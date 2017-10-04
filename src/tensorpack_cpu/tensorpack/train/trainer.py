# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import threading
import time
from six.moves import zip
from tensorflow.python.framework import ops
import numpy as np
from tensorpack.predict.base import DummyOnlinePredictor
from .base import Trainer
import traceback
from ..dataflow.common import RepeatedData

from ..models import TowerContext
from ..utils import *
from ..utils.timer import start_timer, elapsed_time_ms
from ..tfutils import *
from ..tfutils.summary import summary_moving_average, add_moving_summary
from ..tfutils.modelutils import describe_model
from ..predict import OnlinePredictor, build_multi_tower_prediction_graph

import time

__all__ = ['SimpleTrainer', 'QueueInputTrainer']

from tensorflow.python.client import timeline


class PredictorFactory(object):
    """ Make predictors for a trainer"""

    def __init__(self, sess, model, towers, dummy_predictor=False, debug_charts=False):
        """
        :param towers: list of gpu relative id
        """
        self.sess = sess
        self.model = model
        self.towers = towers
        self.tower_built = False
        self.dummy_predictor = dummy_predictor
        self.debug_charts = debug_charts

    def get_predictor(self, input_names, output_names, tower):
        """
        :param tower: need the kth tower (not the gpu id)
        :returns: an online predictor
        """
        if self.dummy_predictor:
            return DummyOnlinePredictor()
        else:
            if not self.tower_built:
                self._build_predict_tower()
            tower = self.towers[tower % len(self.towers)]
            raw_input_vars = get_vars_by_names(input_names)
            output_names = ['towerp{}/'.format(tower) + n for n in output_names]

            #ugly hack by tgrel, forgive me
            output_names.append(GLOBAL_STEP_VAR_NAME)
            output_vars = get_vars_by_names(output_names)
            return OnlinePredictor(self.sess, raw_input_vars, output_vars)


    def _build_predict_tower(self):
        tf.get_variable_scope().reuse_variables()
        # build_predict_tower might get called anywhere, but 'towerp' should be the outermost name scope
        with tf.name_scope(None), \
                freeze_collection(SUMMARY_BACKUP_KEYS):
            build_multi_tower_prediction_graph(
                    self.model, self.towers)
        self.tower_built = True

class SimpleTrainer(Trainer):
    def run_step(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def train(self):
        model = self.model
        self.input_vars = model.get_input_vars()
        with TowerContext(''):
            model.build_graph(self.input_vars)
            cost_var = model.get_cost() # TODO assert scalar
        add_moving_summary(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        grads = self.process_grads(grads)

        avg_maintain_op = summary_moving_average()
        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()
        describe_model()
        # create an infinte data producer
        self.config.dataset.reset_state()

        self.data_producer = RepeatedData(self.config.dataset, -1).get_data()
        self.main_loop()

    def _trigger_epoch(self):
        if self.summary_op is not None:
            data = next(self.data_producer)
            feed = dict(zip(self.input_vars, data))
            summary_str = self.summary_op.eval(feed_dict=feed)
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names):
        if not hasattr(self, 'predictor_factory'):
            self.predictor_factory = PredictorFactory(self.sess, self.model, [0])
        return self.predictor_factory.get_predictor(input_names, output_names, 0)

class EnqueueThread(threading.Thread):
    def __init__(self, trainer):
        super(EnqueueThread, self).__init__()
        self.sess = trainer.sess
        self.coord = trainer.coord

        #tg it uses repeated data source with infinite size
        #what's the config.dataset?
        self.dataflow = RepeatedData(trainer.config.dataset, -1)

        self.input_vars = trainer.input_vars
        self.queue = trainer.input_queue
        self.op = self.queue.enqueue(self.input_vars)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self.size_op = self.queue.size()
        self.daemon = True

    def run(self):
        self.dataflow.reset_state()

        with ops.default_session(self.sess):
            try:
                while True:
                    for dp in self.dataflow.get_data():
                        if self.coord.should_stop():
                            return
                        feed = dict(zip(self.input_vars, dp))
                        self.op.run(feed_dict=feed)
            except tf.errors.CancelledError as e:
                pass
            except Exception:
                logger.exception("Exception in EnqueueThread:")
            finally:
                try:
                    self.sess.run(self.close_op)
                except RuntimeError:    # session already closed
                    pass
                self.coord.request_stop()
                logger.info("Enqueue Thread Exited.")

class QueueInputTrainer(Trainer):
    """ Single GPU Trainer, takes input from a queue"""

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        :param config: a `TrainConfig` instance
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        :param predict_tower: list of gpu relative idx to run prediction. default to be [0].
            Use -1 for cpu.
        """
        super(QueueInputTrainer, self).__init__(config)
        self.input_vars = self.model.get_input_vars()
        # use a smaller queue size for now, to avoid https://github.com/tensorflow/tensorflow/issues/2942


        queue_size = config.extra_arg['queue_size']
        self.dummy_predictor = config.extra_arg['dummy_predictor']
        print 'DUMMY PREDICTOR', self.dummy_predictor

        if input_queue is None:
            self.input_queue = tf.FIFOQueue(
                    queue_size, [x.dtype for x in self.input_vars], name='input_queue')
        else:
            self.input_queue = input_queue

        # by default, use the first training gpu for prediction
        self.predict_tower = predict_tower or [0]
        self.dequed_inputs = None


        self.queue_size_op = self.input_queue.size()

    def _get_model_inputs(self):
        """ Dequeue a datapoint from input_queue and return"""
        #import ipdb; ipdb.set_trace()
        ret = self.input_queue.dequeue(name='input_deque')
        if isinstance(ret, tf.Tensor): # only one input
            ret = [ret]
        assert len(ret) == len(self.input_vars)
        for qv, v in zip(ret, self.input_vars):
            qv.set_shape(v.get_shape())
        return ret

    def _single_tower_grad(self):
        """ Get grad and cost for single-tower"""
        self.dequed_inputs = model_inputs = self._get_model_inputs()

        # test the overhead of queue
        #with tf.device('/gpu:0'):
            #self.dequed_inputs = [tf.Variable(tf.random_normal([128,224,224,3],
                #dtype=tf.float32), trainable=False),
                #tf.Variable(tf.ones([128], dtype=tf.int32), trainable=False)]

        with TowerContext(''):
            self.model.build_graph(self.dequed_inputs)
            cost_var = self.model.get_cost()
        grads = self.config.optimizer.compute_gradients(
                cost_var, gate_gradients=0) # GATE_NONE
        add_moving_summary(cost_var)
        return grads

    def _build_enque_thread(self):
        """ create a thread that keeps filling the queue """
        self.input_th = EnqueueThread(self)
        self._extra_threads_procs.append(self.input_th)

    def train(self):
        assert len(self.config.tower) == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        self.init_session_and_coord()
        self._build_enque_thread()

        grads = self._single_tower_grad()
        grads = self.process_grads(grads)
        describe_model()

        self.train_op = [
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(),
            self.config.model.cost ]

        # skip training
        #self.train_op = tf.group(*self.dequed_inputs)

        self.main_loop()

    def run_step(self):
        """ Simply run self.train_op"""

        idx = 0

        if self.config.extra_arg['threads_to_trace'] > 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            timer = start_timer()

            session_result = self.sess.run(self.train_op, options=run_options, run_metadata=run_metadata)
            #logger.error(session_result)
            self.policy_loss = session_result[2]
            self.xentropy_loss = session_result[3]
            self.value_loss = session_result[4]
            self.grad_norm_before_clip = session_result[5]
            self.grad_norm_after_clip = session_result[6]
            self.active_relus = session_result[7]
            self.advantage = session_result[8]
            self.pred_reward = session_result[9]
            self.step_when_predict = session_result[10]

            global_step = session_result[0]
            if global_step is not None:
                self.global_step = global_step

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
            self.session_result = self.sess.run(self.train_op.op)
            self.session_result = self.train_op.get_output(self.session_result)

            global_step = self.session_result['global_step']
            if global_step is not None:
                self.global_step = global_step

            elapsed_time = elapsed_time_ms(timer)

        try:
            self.delay = self.session_result['delay'] - self.global_step
            self.delay = self.delay * (-1)
            self.delay = self.delay.tolist()
        except Exception as e:
            print "=====EXCEPTION====="
            traceback.print_exc()

        self.elapsed_times[idx].append(elapsed_time)

        #run_metadata = tf.RunMetadata()
        #self.sess.run([self.train_op],
                #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                #run_metadata=run_metadata
                #)
        #from tensorflow.python.client import timeline
        #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #trace_file = open('timeline.ctf.json', 'w')
        #trace_file.write(trace.generate_chrome_trace_format())
        #import sys; sys.exit()

    def _trigger_epoch(self):
        # need to run summary_op every epoch
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names, tower=0):
        """
        :param tower: return the kth predict_func
        :returns: an `OnlinePredictor`
        """
        if not hasattr(self, 'predictor_factory'):
            print 'Creating Predictorfactor', self.dummy_predictor
            self.predictor_factory = PredictorFactory(
                    self.sess, self.model, self.predict_tower, self.dummy_predictor)
        return self.predictor_factory.get_predictor(input_names, output_names, tower)

    def get_predict_funcs(self, input_names, output_names, n):
        return [self.get_predict_func(input_names, output_names, k) for k in range(n)]
