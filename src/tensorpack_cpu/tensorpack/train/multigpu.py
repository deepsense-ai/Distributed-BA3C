#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import copy
import random
from collections import defaultdict

import tensorflow as tf
import itertools, re
from six.moves import zip, range
from tensorflow.python.client import timeline

from ..models import TowerContext
from ..utils import *
from ..utils.timer import start_timer, elapsed_time_ms
from ..utils.concurrency import LoopThread
from ..tfutils.summary import summary_moving_average
from ..tfutils.modelutils import describe_model
from ..tfutils import *

from ..utils.dict_op import TfDictOp

from .trainer import QueueInputTrainer

import os

__all__ = ['AsyncMultiGPUTrainer', 'SyncMultiGPUTrainer']

class MultiGPUTrainer(QueueInputTrainer):
    """ Base class for multi-gpu training"""
    def __init__(self, config, input_queue=None, predict_tower=None):
        super(MultiGPUTrainer, self).__init__(config, input_queue, predict_tower)
        assert len(config.tower) >= 1, "MultiGPUTrainer must be used with at least one GPU."
        self.dequed_inputs = []
        self.dummy = config.extra_arg['dummy']
        print 'MultiGPUTrainer __init__ dummy = {dummy}'.format(dummy=self.dummy)

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                try:
                    grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
                except:
                    logger.error("Error while processing gradients of {}".format(v.name))
                    raise
                ret.append((grad, v))
        return ret

    def _multi_tower_grads(self):
        logger.info("Training a model of {} tower".format(
            len(self.config.tower)))

        grad_list = []
        for idx, t in enumerate(self.config.tower):
            if self.config.extra_arg['cpu'] == 1:
                dev = '/cpu:{}'.format(t)
            else:
                dev = '/gpu:{}'.format(t)
            with tf.device(dev), TowerContext('tower{}'.format(idx)) as scope:
                with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                    logger.info("Building graph for training tower {idx}..., {dev}".format(idx=idx, dev=dev))

                    # IMPORTANT(maciek): Real or Fake?
                    if self.dummy:
                        import numpy as np
                        el1 = tf.ones((128, 84, 84, 12), dtype=tf.float32, name='el1')
                        el2 = tf.constant(np.random.randint(0, 3, size=(128,)), dtype=tf.int32, name='el2')
                        el3 = tf.random_normal((128,), dtype=tf.float32, name='el2')
                        model_inputs = (el1, el2, el3)
                    else:
                        model_inputs = self._get_model_inputs()    # each tower dequeue from input queue

                    self.dequed_inputs.append(model_inputs)
                    #import pdb; pdb.set_trace()
                    self.model.build_graph(model_inputs)
                    cost_var = self.model.get_cost() # build tower

                    # TODO gate_gradienst=0 seems to be faster?
                    grad_list.append(
                        self.config.optimizer.compute_gradients(cost_var, gate_gradients=0))

                    if idx == 0:
                        #tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)
                        tf.get_variable_scope().reuse_variables()
                        # avoid repeated summary from each device
                        backup = backup_collection(SUMMARY_BACKUP_KEYS)

        restore_collection(backup)
        return grad_list

class Ref(object):
    def __init__(self, wrapped=None):
        self.wrapped = wrapped

    def set(self, wrapped):
        self.wrapped = wrapped

    def get(self):
        return self.wrapped

    def __getattr__(self, name):
        return self.wrapped.__getattribute__(name)

class SyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.init_session_and_coord()
        self.elapsed_times = defaultdict(list)
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()

        grads = MultiGPUTrainer._average_grads(grad_list)
        grads = self.process_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        describe_model()

        # [debug]: do nothing in training
        #self.train_op = self.dequed_inputs[0][0] + self.dequed_inputs[1][0]
        self.main_loop()


class AsyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.sess = Ref()
        self.coord = Ref()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()

        # pretend to average the grads, in order to make async and
        # sync have consistent effective learning rate
        def scale(grads):
            with tf.name_scope('AsyncScaleGrad'):
                return [(grad / self.config.extra_arg['n_workers'] if grad is not None else None, var)
                            for grad, var in grads]

        def flatten_list_of_lists(lists):
            flattened_list = []
            for l in lists:
                flattened_list.extend(l)
            return flattened_list


        grad_norms_before_process = map(lambda grads: [tf.norm(g) for g, v in grads], grad_list)
        grad_norms_before_process = tf.sqrt(tf.add_n([tf.square(g) for g in flatten_list_of_lists(grad_norms_before_process)]))

        #grad_list = map(scale, grad_list)
        grad_list = [self.process_grads(g) for g in grad_list]

        grad_norms_after_process = map(lambda grads: [tf.norm(g) for g, v in grads], grad_list)
        grad_norms_after_process = tf.sqrt(tf.add_n([tf.square(g) for g in flatten_list_of_lists(grad_norms_after_process)]))

        # use grad from the first tower for iteration in main thread
        with tf.device(self.config.extra_arg['device_function']):
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                global_step_var = get_global_step_var()

        if self.config.extra_arg['debug_charts']:
            op_dict = {
                'global_step' : self.config.optimizer.apply_gradients(grad_list[0], global_step_var),
                'summary' : summary_moving_average(),
                'cost' : self.config.model.cost,
                'policy_loss' : self.config.model.policy_loss,
                'xentropy_loss' : self.config.model.xentropy_loss,
                'value_loss' : self.config.model.value_loss,
                'grad_norm_before_process' : grad_norms_before_process,
                'grad_norm_after_process' : grad_norms_after_process,
                'active_relus' : self.config.model.active_relus,
                'advantage' : self.config.model.advantage,
                'pred_reward' : self.config.model.pred_reward,
                'delay' : self.config.model.delay,
                'mean_action' : self.config.model.mean_action,
                'mean_future_reward' : self.config.model.mean_future_reward,
                'mean_state' : self.config.model.mean_state,
                'mean_init_R' : self.config.model.mean_init_R,
                'games_over' : self.config.model.games_over,
                'max_logit' : self.config.model.max_logit,
                'mean_value' : self.config.model.mean_value,
                'fc_fc0' : self.config.model.fc_fc0,
                'fc_value' : self.config.model.fc_value
                }

        else: # not self.config.extra_arg['debug_charts']
            op_dict = {
                    'global_step' : self.config.optimizer.apply_gradients(grad_list[0], global_step_var),
                    'summary' : summary_moving_average(),
                    'cost' : self.config.model.cost,
                    'policy_loss' : self.config.model.policy_loss,
                    'xentropy_loss' : self.config.model.xentropy_loss,
                    'value_loss' : self.config.model.value_loss,
                    'advantage' : self.config.model.advantage,
                    'pred_reward' : self.config.model.pred_reward,
                    'delay' : self.config.model.delay,
                    'active_relus' : self.config.model.active_relus,
                    'max_logit' : self.config.model.max_logit
                    }

        if self.config.extra_arg['adam_debug']:
            op_dict['adam_m_norm'] = self.config.optimizer.get_m_norm()
            op_dict['adam_v_norm'] = self.config.optimizer.get_v_norm()
            op_dict['adam_update_norm'] = self.config.optimizer.get_update_norm()
            op_dict['adam_lr'] = self.config.optimizer.get_lr_adapt()

        if self.config.extra_arg['schedule_hyper']:
            op_dict['learning_rate'] = self.config.extra_arg['lr']
            op_dict['entropy_beta'] = self.config.model.entropy_beta

        self.train_op = TfDictOp(op_dict)
        describe_model()

        self._start_async_threads(grad_list)
        self.step_counter = 0
        self.main_thread_timer = start_timer()
        self.main_thread_counter = 0
        self.step_times = []

        self._init_summary()
        get_global_step_var()   # ensure there is such var, before finalizing the graph
        logger.info("Setup callbacks ...")
        callbacks = self.config.callbacks
        callbacks.setup_graph(self) # TODO use weakref instead?
        self.init_session_and_coord()
        self.main_loop()


    def _start_async_threads(self, grad_list):
        import threading
        # prepare train_op for the rest of the towers
        # itertools.count is atomic w.r.t. python threads
        self.async_step_counter = itertools.count()
        self.training_threads = []


        queue_size_op = self.input_queue.size()
        self.elapsed_times = defaultdict(list)

        for k in range(1, len(self.config.tower)):
            train_op = self.config.optimizer.apply_gradients(grad_list[k])

            def f(op=train_op, idx=k): # avoid late-binding
                oper_id = random.randint(0, 10000)
                queue_size = self.sess.run([queue_size_op])
                print 'Queueu_size', queue_size
                if idx <= self.config.extra_arg['threads_to_trace']:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    timer = start_timer()
                    self.sess.run([op], options=run_options, run_metadata=run_metadata)
                    elapsed_time = elapsed_time_ms(timer)
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline_intra_{intra_op_par}_nrthreads_{nr_threads}_thidx_{th_idx}_{oper_id}.json'.format(
                            intra_op_par=self.config.extra_arg['intra_op_par'],
                            #inter_op_par=self.config.extra_arg['inter_op_par'],
                            nr_threads=len(self.config.tower),
                            th_idx=idx,
                            oper_id=len(self.elapsed_times[idx])), 'w') as f:
                        f.write(ctf)
                else:
                    timer = start_timer()
                    self.sess.run([op])
                    elapsed_time = elapsed_time_ms(timer)


                self.elapsed_times[idx].append(elapsed_time)
                #print 'completed', idx, oper_id
                next(self.async_step_counter)

            th = LoopThread(f)
            th.pause()
            th.start()
            self.training_threads.append(th)
        self.async_running = False

    def run_step(self):
        if not self.async_running:
            self.async_running = True
            for th in self.training_threads: # resume all threads
                th.resume()
        next(self.async_step_counter)

        if self.config.extra_arg['max_steps'] is not None and self.step_counter >= self.config.extra_arg['max_steps']:
            import os
            import signal
            os.killpg(os.getpgrp(), signal.SIGKILL)

            import sys
            sys.exit()
        else:
            self.step_counter += 1

        s = ("Q-debug id=dkjs, tf_queue_size {qsize}".format(qsize=self.queue_size_op.eval()))
        logger.debug(s)

        super(AsyncMultiGPUTrainer, self).run_step()

        self.main_thread_counter += 1
        elapsed_time = elapsed_time_ms(self.main_thread_timer)
        self.step_times.append(elapsed_time)
        last_k = 20
        mean_step_time = np.mean(self.step_times[-last_k:])

        self.dp_per_s = 1000.0 / mean_step_time * self.config.extra_arg['batch_size']

        #if int(self.async_step_counter.__str__()) % 100 == 0:
        import os
        s = ("[{node}]  step: {step}, step_time {step_time}, mean_step_time {mean_step_time}, it/s {it_s}".format(
             node=os.getenv("SLURMD_NODENAME", "none"),
             step=self.async_step_counter.__str__(),
             step_time=round(elapsed_time, 2),
             mean_step_time=round(mean_step_time,2),
             it_s=round(1000.0 / mean_step_time, 2)))
        logger.error(s)
        self.main_thread_timer = start_timer()

    def _trigger_epoch(self):
        self.async_running = False
        for th in self.training_threads:
            th.pause()
        try:
            async_step_total_cnt = int(re.findall(
                '[0-9]+', self.async_step_counter.__str__())[0])
            self.write_scalar_summary(
                    'async_global_step', async_step_total_cnt)
        except:
            pass
        super(AsyncMultiGPUTrainer, self)._trigger_epoch()
