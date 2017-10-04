#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import random, time
import threading, multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue
import traceback
from tensorpack import *
from tensorpack.predict import get_predict_func
from tensorpack.utils.concurrency import *
from tensorpack.utils.stat import  *
from tensorpack.callbacks import *
from tensorpack.callbacks.base import ProxyCallback

import sys
import os

global get_player
get_player = None

def play_one_episode(player, func, verbose=False):
    def f(s):
        spc = player.get_action_space()
        act = func([[s]])[0][0].argmax()
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act
    return np.mean(player.play_one_episode(f))

def play_model(cfg):
    player = get_player(viz=0.01)
    predfunc = get_predict_func(cfg)
    while True:
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def eval_with_funcs(predict_funcs, nr_eval):
    class Worker(StoppableThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            player = get_player(train=False)
            while not self.stopped():
                try:
                    score = play_one_episode(player, self.func)
                    #print "Score, ", score
                except RuntimeError:
                    return
                self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predict_funcs]

    for k in threads:
        k.start()
        time.sleep(0.1) # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            r = q.get()
            stat.feed(r)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads: k.stop()
        for k in threads: k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        if stat.count > 0:
            return (stat.average, stat.max)
        return (0, 0)

def eval_model_multithread(cfg, nr_eval):
    func = get_predict_func(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))

class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, neptune_client, worker_id, solved_score=None):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.neptune_client = neptune_client
        self.worker_id = worker_id
        self.solved_score = solved_score

        # neptunization debugging, to be removed in the future
        if False:
            def thread_func(neptune_client, worker_id):
                while True:
                    time.sleep(1)
                    print 'sending points...'
                    content = ('score', -2.1, -1.1)
                    message = (worker_id, content)
                    neptune_client.send(message)

            self.t = threading.Thread(target=thread_func, args=[neptune_client, worker_id])
            self.t.start()

    def _setup_graph(self):
        try:
            NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
            self.pred_funcs = [self.trainer.get_predict_func(
                self.input_names, self.output_names)] * NR_PROC
        except Exception as e:
            print ('============= EXCEPTION WHEN CREATING THE EVALUATION GRAPH [{}]======== '.format(os.environ["SLURMD_NODENAME"]))
            traceback.print_exc()

    def _trigger_epoch(self):
        t = time.time()
        mean, max = eval_with_funcs(self.pred_funcs, nr_eval=self.eval_episode)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.write_scalar_summary('mean_score', mean)
        self.trainer.write_scalar_summary('max_score', max)

        content = ('score', mean, max)
        message = (self.worker_id, content)
        print 'sending points...'
        self.neptune_client.send(message)

        if self.solved_score is not None:
            if mean >= self.solved_score:
                os.system('scancel {}'.format(os.environ['SLURM_JOB_ID']))

class HeartPulseCallback(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.fd = open(self.filename, 'w')

        self.fd.write('{}\n'.format(os.environ['SLURMD_NODENAME']))
        self.fd.flush()

        self.step_counter = 0

    def trigger_step(self):
        self.step_counter += 1

        self.fd.write('step {}: alive\n'.format(self.step_counter))
        self.fd.flush()

class PeriodicPerStepCallback(ProxyCallback):
    def __init__(self, cb, n):

        super(PeriodicPerStepCallback, self).__init__(cb)
        self.n = n
        self.counter = 0

    def trigger_step(self):
        self.counter += 1
        if self.counter == self.n:
            self.cb.trigger_step()
            self.counter = 0

class DebugLogCallback(Callback):
    def __init__(self, neptune_client, worker_id, nr_send,
            debug_charts, adam_debug, schedule_hyper):
        self.neptune_client = neptune_client
        self.worker_id = worker_id
        self.send_every = nr_send
        self.debug_charts = debug_charts
        self.adam_debug = adam_debug
        self.schedule_hyper = schedule_hyper

        self.dp_per_s = StatCounter()
        self.step_counter = 0
        self.counter = 0

    def trigger_step(self):
        if not hasattr(self, 'lists'):
            self.lists = {
                    name : StatCounter() for name in self.trainer.session_result
                    }

        for name in self.trainer.session_result:
            self.lists[name].feed(self.trainer.session_result[name])
        self.dp_per_s.feed(self.trainer.dp_per_s)

        self.counter += 1
        self.step_counter += 1
        if self.counter == self.send_every:
            loss_content = ('loss',
                    float(self.lists['cost'].average),
                    float(self.lists['policy_loss'].average),
                    float(self.lists['xentropy_loss'].average),
                    float(self.lists['value_loss'].average),
                    float(self.lists['advantage'].average),
                    float(self.lists['pred_reward'].average),
                    float(self.lists['max_logit'].average)
                    )
            other_content = ('other',
                    float(self.lists['active_relus'].average),
                    float(self.dp_per_s.average)
                    )

            if self.adam_debug:
                adam_content = ('adam',
                        float(self.lists['adam_m_norm'].average),
                        float(self.lists['adam_v_norm'].average),
                        float(self.lists['adam_update_norm'].average),
                        float(self.lists['adam_lr'].average)
                        )

            if self.schedule_hyper:
                schedule_content = ('schedule',
                        float(self.lists['learning_rate'].average),
                        float(self.lists['entropy_beta'].average)
                        )

            if self.debug_charts:
                grad_content = ('grad',
                        float(self.lists['grad_norm_before_process'].average),
                        float(self.lists['grad_norm_after_process'].average)
                        )
                other_content += (
                        float(self.lists['fc_fc0'][0]),
                        float(self.step_counter / self.send_every),
                        float(self.lists['fc_value'][0])
                        )
                state_content = ('env_state',
                        float(self.lists['mean_state'].average),
                        float(self.lists['mean_future_reward'].average),
                        float(self.lists['mean_value'].average),
                        float(self.lists['mean_init_R'].average),
                        float(self.lists['mean_action'].average),
                        float(self.lists['games_over'].average)
                        )

            loss_message = (self.worker_id, loss_content)
            other_message = (self.worker_id, other_content)

            if self.adam_debug:
                adam_message = (self.worker_id, adam_content)

            if self.schedule_hyper:
                schedule_message = (self.worker_id, schedule_content)

            if self.debug_charts:
                grad_message = (self.worker_id, grad_content)
                state_message = (self.worker_id, state_content)

            print 'sending debugging info...'
            try:
                self.mean_delays = np.mean(self.trainer.delay)
                self.max_delays = np.max(self.trainer.delay)
                self.min_delays = np.min(self.trainer.delay)
                delays_content = ('delays', (self.mean_delays, self.max_delays, self.min_delays))
                delays_message = (self.worker_id, delays_content)
                self.neptune_client.send(delays_message)
            except Exception as e:
                print ' EXCEPTION WHILE SENDING DELAYS :('
                print traceback.format_exc(e)

            self.neptune_client.send(loss_message)
            self.neptune_client.send(other_message)

            if self.adam_debug:
                self.neptune_client.send(adam_message)

            if self.schedule_hyper:
                self.neptune_client.send(schedule_message)

            if self.debug_charts:
                self.neptune_client.send(grad_message)
                self.neptune_client.send(state_message)

            # reset counter and lists
            self.counter = 0
            for name in self.lists:
                self.lists[name].reset()
            self.dp_per_s.reset()

class HyperParameterScheduler(Callback):
    def __init__(self, name, schedule):
        _, self.name = get_op_var_name(name)
        self.schedule = schedule

    def _setup_graph(self):
        all_vars = tf.global_variables()
        for v in all_vars:
            if v.name == self.name:
                self.var = v
                return
        self.var = None

    def _trigger_epoch(self):
        if len(self.schedule) > 0:
            if self.var is not None:
                for e, v in self.schedule:
                    if e == self.epoch_num:
                        self.var.load(v)

