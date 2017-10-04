#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from datetime import datetime
import numpy as np
import time
import uuid
from six.moves import queue
from tensorflow.python.framework import ops
from tensorpack.utils.timer import start_timer, elapsed_time_ms
from tensorpack import *
from tensorpack.RL.common import MapPlayerState
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.RL.simulator import SimulatorMaster, SimulatorProcess
from tensorpack.callbacks.stat import StatPrinter
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.raw import DataFromQueue
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.model_desc import get_current_tower_context
from tensorpack.models.pool import MaxPooling
from tensorpack.predict.common import PredictConfig
from tensorpack.predict.concurrency import MultiThreadAsyncPredictor
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.config import TrainConfig
from tensorpack.train.multigpu import AsyncMultiGPUTrainer, SyncMultiGPUTrainer
from tensorpack.utils.concurrency import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.serialize import *
from tensorpack.utils.stat import StatCounter
from tensorpack.tfutils import symbolic_functions as symbf
from tensorflow_slurm_utils import tf_server_from_slurm
from tensorpack.RL import *
import common
from common import (play_model, Evaluator, HyperParameterScheduler, PeriodicPerStepCallback, DebugLogCallback, HeartPulseCallback, eval_model_multithread)
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import traceback
import socket
import sys

from collections import defaultdict

# for SEGFAULT debugging
import neptune_mp_server
import sys
import os

sys.path.append('.')

# parse args
import parse
args = parse.parse_args()
for arg in vars(args):
    if getattr(args, arg) == "None":
        setattr(args, arg, None)

def eprint(*args, **kwargs):
    sys.stderr.write(','.join(args))

cluster, my_job_name, my_task_index = tf_server_from_slurm(ps_number=args.ps, port_number=args.tf_port)
if args.eval_node:
    cluster['worker'] = cluster['worker'][:-1]
if args.record_node:
    cluster['worker'] = cluster['worker'][:-1]
cluster_spec = tf.train.ClusterSpec(cluster)
print(cluster)

retries = 10
for i in range(retries):
    try:
        print('[{}:{}] Starting the TF server'.format(my_job_name, my_task_index))
        server = tf.train.Server(cluster_spec, job_name=my_job_name, task_index=my_task_index)
        break
    except Exception as e:
        print '========= EXCEPTION WHILE STARTING TF SERVER [{}] ====='.format(os.environ['SLURMD_NODENAME'])
        traceback.print_exc()
        time.sleep(1)

if my_job_name == 'ps':
    logger._logger.disabled = 0
    logger.info('[{}:{}] joining the server.'.format(my_job_name, my_task_index))
    server.join()
    os.environ['FINISHED'] = '1'
    exit(0)
is_chief = my_task_index == 0

logger._logger.disabled = 0

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * args.channels
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

## ugly hack:
TARGET_CHANNELS = 16
#IMAGE_SHAPE3 = IMAGE_SIZE + (TARGET_CHANNELS,)

LOCAL_TIME_MAX = 5
STEP_PER_EPOCH = args.steps_per_epoch
EVAL_EPISODE = 50

BATCH_SIZE = args.batch_size

NUM_ACTIONS = None
ENV_NAME = None

#PARAMETER_SERVER_DEVICE = "/job:ps/task:0"

def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)

    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1])

    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl


common.get_player = get_player


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    def __init__(self, device_function):
        self.device_function = device_function
        self.vars_for_save = {}

    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'futurereward'),
                InputVar(tf.float32, (args.batch_size,), 'global_step_from_predict'),
                InputVar(tf.float32, (None,), 'init_R'),
                InputVar(tf.bool, (None,), 'isOver')]

    def _get_NN_prediction(self, image, retry=False):
        if retry:
            while True:
                try:
                    res = self._get_NN_prediction_wrapped(image)
                except InvalidArgumentError as e:
                    pass
                    time.sleep(1)
        else:
            return self._get_NN_prediction_wrapped(image)

    def _get_NN_prediction_wrapped(self, image):
        with tf.device('/cpu:0'):
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                image = image / 255.0

                relus = []
                self.layer_output_means = {}

                print(CHANNEL)
                dummy_channels = TARGET_CHANNELS - CHANNEL
                image = tf.concat([image, tf.zeros((tf.shape(image)[0], 84, 84, dummy_channels))], 3)
                with argscope(Conv2D, nl=tf.nn.relu):
                    input_sum = tf.reduce_sum(tf.abs(image))
                    l = Conv2D('conv0', image, out_channel=32, kernel_shape=5, use_bias=False, padding='VALID',
                               parameter_device=self.device_function, conv_init=args.conv_init)
                    with tf.variable_scope('conv0', reuse=True):
                        self.vars_for_save['conv0/W'] = tf.get_variable('W')

                    self.layer_output_means['conv0_out'] = tf.reduce_mean(l)
                    relus.append(l)
                    # l = tf.layers.batch_normalization(l)
                    l = MaxPooling('pool0', l, 2)

                    l = Conv2D('conv1', l, out_channel=32, kernel_shape=5, use_bias=False, padding='VALID',
                               parameter_device=self.device_function, conv_init=args.conv_init)
                    with tf.variable_scope('conv1', reuse=True):
                        self.vars_for_save['conv1/W'] = tf.get_variable('W')

                    self.layer_output_means['conv1_out'] = tf.reduce_mean(l)
                    relus.append(l)
                    # l = tf.Print(l, [l], message='conv1: ', summarize=30)
                    l = MaxPooling('pool1', l, 2)

                    l = Conv2D('conv2', l, out_channel=64, kernel_shape=5, use_bias=False, padding='VALID',
                               parameter_device=self.device_function, conv_init=args.conv_init)
                    with tf.variable_scope('conv2', reuse=True):
                        self.vars_for_save['conv2/W'] = tf.get_variable('W')

                    self.layer_output_means['conv2_out'] = tf.reduce_mean(l)
                    relus.append(l)
                    l = MaxPooling('pool2', l, 2)

                    l = Conv2D('conv3', l, out_channel=64, kernel_shape=3, use_bias=False, padding='VALID',
                               parameter_device=self.device_function, conv_init=args.conv_init)
                    with tf.variable_scope('conv3', reuse=True):
                        self.vars_for_save['conv3/W'] = tf.get_variable('W')

                    self.layer_output_means['conv3_out'] = tf.reduce_mean(l)
                    relus.append(l)

                # 1 fully connected layer but split into many tensors
                # to better split parameters between many PS's
                if args.replace_with_conv:
                    fc_splits = []
                    neurons = args.fc_neurons / args.fc_splits
                    for i in range(args.fc_splits):
                        fc = Conv2D('fc1_{}'.format(i), l, out_channel=neurons, kernel_shape=5,
                                     nl=tf.identity, padding='VALID', parameter_device=self.device_function, conv_init='uniform2',
                                     use_bias=False)
                        with tf.variable_scope('fc1_{}'.format(i), reuse=True):
                            self.vars_for_save['fc1_{}/W'.format(i)] = tf.get_variable('W')

                        fc = tf.reshape(tensor=fc, shape=[-1, neurons])
                        self.layer_output_means['fc1_{}_out'.format(i)] = tf.reduce_mean(fc)
                        fc_splits.append(fc)
                    l = tf.concat(fc_splits, axis=1)
                else:
                    fc = []
                    for i in range(args.ps):
                        fc_part = FullyConnected('fc1_{}'.format(i), l, args.fc_neurons / args.ps, nl=tf.identity,
                            parameter_device=self.device_function, fc_init=args.fc_init)
                        with tf.variable_scope('fc1_{}'.format(i), reuse=True):
                            self.vars_for_save['fc1_{}/W'.format(i)] = tf.get_variable('W')
                            self.vars_for_save['fc1_{}/b'.format(i)] = tf.get_variable('b')

                        self.layer_output_means['fc1_{}_out'.format(i)] = tf.reduce_mean(fc_part)
                        fc_part = tf.nn.relu(fc_part, 'relu1')
                        fc.append(fc_part)
                        relus.append(fc_part)
                    l = tf.concat(fc, axis=1)

                with tf.variable_scope('fc1_0', reuse=True):
                    val = tf.gather(tf.get_variable('W'), 0)
                    val = tf.reshape(val, [-1])
                    self.fc_fc0 = tf.gather(val, 0)

                policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity,
                                        parameter_device=self.device_function, fc_init=args.fc_init)
                with tf.variable_scope('fc-pi', reuse=True):
                    self.vars_for_save['fc-pi/W'] = tf.get_variable('W')
                    self.vars_for_save['fc-pi/b'] = tf.get_variable('b')

                self.layer_output_means['fc_pi_out'] = tf.reduce_mean(policy)

                value = FullyConnected('fc-v', l, out_dim=1, nl=tf.identity,
                                       parameter_device=self.device_function, fc_init=args.fc_init)
                with tf.variable_scope('fc-v', reuse=True):
                    self.vars_for_save['fc-v/W'] = tf.get_variable('W')
                    self.vars_for_save['fc-v/b'] = tf.get_variable('b')

                self.layer_output_means['fc_v_out'] = tf.reduce_mean(value)

                with tf.variable_scope('fc-v', reuse=True):
                    val = tf.gather(tf.get_variable('W'), 0)
                    self.fc_value = tf.gather(val, 0)

                # number of relu gates that are not 0
                self.active_relus = tf.add_n([tf.count_nonzero(r) for r in relus])
                return policy, value

    def _build_graph(self, inputs):
        # import ipdb; ipdb.set_trace()
        eprint('===== [{}] PRINTING BUILD GRAPH STACK AT {}=============='.format(socket.gethostname(),
                                                                                  time.time()))
        traceback.print_stack(file=sys.stderr)


        with tf.device('/cpu:0'):
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                state, action, futurereward, global_step_from_predict, init_R, isOver = inputs
                self.mean_state = tf.reduce_mean(state)
                self.mean_action = tf.reduce_mean(tf.cast(action, tf.float32))
                self.mean_future_reward = tf.reduce_mean(futurereward)
                self.mean_init_R = tf.reduce_mean(init_R)
                self.games_over = tf.reduce_sum(tf.cast(isOver, tf.float32))

                self.delay = global_step_from_predict
                policy, self.value = self._get_NN_prediction(state)
                self.value = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
                self.logits = tf.nn.softmax(policy, name='logits')

                self.mean_value = tf.reduce_mean(self.value, name='mean_value')
                self.max_logit = tf.reduce_max(self.logits, name='max_logit')

                with tf.device(self.device_function):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                        self.expf = tf.get_variable('explore_factor', shape=[],
                                               initializer=tf.constant_initializer(1), trainable=False)
                        self.entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                                       initializer=tf.constant_initializer(0.01), trainable=False)

                batch_size = tf.cast(tf.shape(futurereward)[0], tf.float32)

                logitsT = tf.nn.softmax(policy * self.expf, name='logitsT')
                is_training = get_current_tower_context().is_training
                if not is_training:
                    return
                log_probs = tf.log(self.logits + 1e-6)

                log_pi_a_given_s = tf.reduce_sum(
                    log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
                advantage = tf.subtract(tf.stop_gradient(self.value), futurereward, name='advantage')
                policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage, name='policy_loss')
                self.policy_loss = policy_loss * 128 / batch_size
                # policy_loss = tf.Print(policy_loss, [policy_loss], 'policy_loss')

                xentropy_loss = tf.reduce_sum(
                    self.logits * log_probs, name='xentropy_loss')

                self.xentropy_loss = xentropy_loss * 128 / batch_size
                self.xentropy_loss *= self.entropy_beta

                value_loss = tf.nn.l2_loss(self.value - futurereward, name='value_loss')
                self.value_loss = value_loss * 128 / batch_size
                self.pred_reward = tf.reduce_mean(self.value, name='predict_reward')

                self.advantage = tf.reduce_mean(advantage)

                self.cost = tf.add_n([policy_loss, xentropy_loss * self.entropy_beta, value_loss])
                self.cost = tf.truediv(self.cost, batch_size, name='cost')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1))]
                # SummaryGradient()]

class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, worker_id, neptune_client, pipe_c2s, pipe_s2c, model, dummy, predictor_threads, predict_batch_size=16, do_train=True):
        # predictor_threads is previous PREDICTOR_THREAD
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c, args.simulator_procs, os.getpid())
        self.M = model
        self.do_train = do_train

        # the second queue is here!
        self.queue = queue.Queue(maxsize=args.my_sim_master_queue)
        self.dummy = dummy
        self.predictor_threads = predictor_threads

        self.last_queue_put = start_timer()
        self.queue_put_times = []
        self.predict_batch_size = predict_batch_size
        self.counter = 0

        self.worker_id = worker_id
        self.neptune_client = neptune_client
        self.stats = defaultdict(StatCounter)
        self.games = StatCounter()

    def _setup_graph(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                self.sess = self.trainer.sess
                self.async_predictor = MultiThreadAsyncPredictor(
                    self.trainer.get_predict_funcs(['state'], ['logitsT', 'pred_value'], self.predictor_threads),
                    batch_size=self.predict_batch_size)
                self.async_predictor.run()

    def _on_state(self, state, ident):
        ident, ts = ident
        client = self.clients[ident]

        if self.dummy:
            action = 0
            value = 0.0
            client.memory.append(TransitionExperience(state, action, None, value=value))
            self.send_queue.put([ident, dumps(action)])
        else:
            def cb(outputs):
                # distrib, value, global_step, isAlive  = outputs.result()
                o = outputs.result()
                if o[-1]:
                    distrib = o[0]
                    value = o[1]
                    global_step = o[2]
                    assert np.all(np.isfinite(distrib)), distrib
                    action = np.random.choice(len(distrib), p=distrib)
                    client = self.clients[ident]
                    client.memory.append(TransitionExperience(state, action, None, value=value, ts=ts))
                else:
                    self.send_queue.put([ident, dumps((0, 0, False))])
                    return

                #print"Q-debug: MySimulatorMaster send_queue before put, size: ", self.send_queue.qsize(), '/', self.send_queue.maxsize
                self.send_queue.put([ident, dumps((action, global_step, True))])

            self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        ident, ts = ident

        client = self.clients[ident]
        # send game score to neptune
        self.games.feed(self.stats[ident].sum)
        self.stats[ident].reset()

        if self.games.count == 10:
            self.neptune_client.send((self.worker_id, ('online', self.games.average)))
            self.games.reset()

        self._parse_memory(0, ident, True, ts)

    def _on_datapoint(self, ident):
        ident, ts = ident
        client = self.clients[ident]

        self.stats[ident].feed(client.memory[-1].reward)

        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False, ts)

    def _parse_memory(self, init_r, ident, isOver, ts):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            point_ts = k.ts
            self.log_queue_put()
            if self.do_train:
                self.queue.put([k.state, k.action, R, point_ts, init_r, isOver])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def log_queue_put(self):
        self.counter += 1
        elapsed_last_put = elapsed_time_ms(self.last_queue_put)
        self.queue_put_times.append(elapsed_last_put)
        k = 1000
        if self.counter % 1 == 0:
            logger.debug("queue_put_times elapsed {elapsed}".format(elapsed=elapsed_last_put))
            logger.debug("queue_put_times {puts_s} puts/s".format(puts_s=1000.0 / np.mean(self.queue_put_times[-k:])))
        self.last_queue_put = start_timer()


def _chief_worker_hostname(cluster):
    return cluster['worker'][0].split(':')[0]

class MyAdamOptimizer(tf.train.AdamOptimizer):
    '''
    class for getting internal state of AdamOptimizer
    '''
    def __init__(self, learning_rate=0.001, beta1=0.9,
            beta2=0.999, epsilon=1e-08, use_locking=False,
            name='Adam'):
        super(MyAdamOptimizer, self).__init__(
                learning_rate, beta1, beta1,  epsilon, use_locking, name)

        self.m_hat = {}
        self.v_hat = {}
        self.update = {}

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        self.m_hat[var.name] = m/(1 - self._beta1_power)
        self.v_hat[var.name] = v/(1 - self._beta2_power)
        self.update[var.name] = self._lr_t * self.m_hat[var.name] / (tf.sqrt(self.v_hat[var.name]) + self._epsilon_t)

        if not hasattr(self, 'lr_adapt'):
            self.lr_adapt = (self._lr_t * tf.sqrt(1 - self._beta2_power) / (1 - self._beta1_power))

        # do actual update
        return super(MyAdamOptimizer, self)._apply_dense(grad, var)

    def get_lr_adapt(self):
        return self.lr_adapt

    def get_m_norm(self):
        m_square_sum = tf.add_n([tf.reduce_sum(tf.square(self.m_hat[val])) for val in self.m_hat])
        return tf.sqrt(m_square_sum)

    def get_v_norm(self):
        v_square_sum = tf.add_n([tf.reduce_sum(tf.square(self.v_hat[val])) for val in self.v_hat])
        return tf.sqrt(v_square_sum)

    def get_update_norm(self):
        update_square_sum = tf.add_n([tf.reduce_sum(tf.square(self.update[val])) for val in self.update])
        return tf.sqrt(update_square_sum)

class MySyncReplicasOptimizer(tf.train.SyncReplicasOptimizer):
    def get_lr_adapt(self):
        return self._opt.get_lr_adapt()

    def get_m_norm(self):
        return self._opt.get_m_norm()

    def get_v_norm(self):
        return self._opt.get_v_norm()

    def get_update_norm(self):
        return self._opt.get_update_norm()

def get_config(args=None, is_chief=True, task_index=0, chief_worker_hostname="", n_workers=1):
    logger.set_logger_dir(args.train_log_path + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(task_index))

    # function to split model parameters between multiple parameter servers
    ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(cluster['ps']), tf.contrib.training.byte_size_load_fn)
    device_function = tf.train.replica_device_setter(worker_device='/job:worker/task:{}/cpu:0'.format(task_index),
                                                    cluster=cluster_spec, ps_strategy=ps_strategy)

    M = Model(device_function)

    name_base = str(uuid.uuid1()).replace('-', '')[:16]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(args.simulator_procs)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    neptune_client = neptune_mp_server.Client(server_host=chief_worker_hostname, server_port=args.port)

    master = MySimulatorMaster(task_index, neptune_client, namec2s, names2c, M, dummy=args.dummy,
                               predictor_threads=args.nr_predict_towers, predict_batch_size=args.predict_batch_size,
                               do_train=args.do_train)

    # here's the data passed to the repeated data source
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)

    with tf.device(device_function):
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)

    intra_op_par = args.intra_op_par
    inter_op_par = args.inter_op_par

    session_config = get_default_sess_config(0.5)
    print("{} {}".format(intra_op_par, type(intra_op_par)))
    if intra_op_par is not None:
        session_config.intra_op_parallelism_threads = intra_op_par

    if inter_op_par is not None:
        session_config.inter_op_parallelism_threads = inter_op_par

    session_config.log_device_placement = False
    extra_arg = {
        'dummy_predictor': args.dummy_predictor,
        'intra_op_par': intra_op_par,
        'inter_op_par': inter_op_par,
        'max_steps': args.max_steps,
        'device_count': {'CPU': args.cpu_device_count},
        'threads_to_trace': args.threads_to_trace,
        'dummy': args.dummy,
        'cpu' : args.cpu,
        'queue_size' : args.queue_size,
        #'worker_host' : "grpc://localhost:{}".format(cluster['worker'][my_task_index].split(':')[1]),
        'worker_host' : server.target,
        'is_chief' : is_chief,
        'device_function': device_function,
        'n_workers' : n_workers,
        'use_sync_opt' : args.use_sync_opt,
        'port' : args.port,
        'batch_size' : BATCH_SIZE,
        'debug_charts' : args.debug_charts,
        'adam_debug' : args.adam_debug,
        'task_index' : task_index,
        'lr' : lr,
        'schedule_hyper' : args.schedule_hyper,
        'experiment_dir' : args.experiment_dir
    }

    print("\n\n worker host: {} \n\n".format(extra_arg['worker_host']))


    with tf.device(device_function):
        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(lr, epsilon=args.epsilon, beta1=args.beta1, beta2=args.beta2)
            if args.adam_debug:
                optimizer = MyAdamOptimizer(lr, epsilon=args.epsilon, beta1=args.beta1, beta2=args.beta2)
        elif args.optimizer == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif args.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(lr)
        elif args.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(lr, epsilon=1e-3)
        elif args.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif args.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(lr)

        # wrap in SyncReplicasOptimizer
        if args.use_sync_opt == 1:
            if not args.adam_debug:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=args.num_grad,
                                                            total_num_replicas=n_workers)
            else:
                optimizer = MySyncReplicasOptimizer(optimizer, replicas_to_aggregate=args.num_grad,
                                                            total_num_replicas=n_workers)
            extra_arg['hooks'] = optimizer.make_session_run_hook(is_chief)

    callbacks = [
            StatPrinter(),
            master,
            DebugLogCallback(neptune_client, worker_id=task_index, nr_send=args.send_debug_every,
                            debug_charts=args.debug_charts, adam_debug=args.adam_debug, schedule_hyper=args.schedule_hyper)
           ]

    if args.debug_charts:
        callbacks.append(HeartPulseCallback('heart_pulse_{}.log'.format(os.environ['SLURMD_NODENAME'])))

    if args.early_stopping is not None:
        args.early_stopping = float(args.early_stopping)

        if my_task_index == 1 and not args.eval_node:
            # only one worker does evaluation
            callbacks.append(PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['logits'], neptune_client, worker_id=task_index,
                solved_score=args.early_stopping), 2))
    elif my_task_index == 1 and not args.eval_node:
        # only 1 worker does evaluation
        callbacks.append(PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['logits'], neptune_client, worker_id=task_index), 2))

    if args.save_every != 0:
        callbacks.append(PeriodicPerStepCallback(ModelSaver(var_collections=M.vars_for_save, models_dir=args.models_dir), args.save_every))

    if args.schedule_hyper and my_task_index == 2:
        callbacks.append(HyperParameterScheduler('learning_rate', [(20, 0.0005), (60, 0.0001)]))
        callbacks.append(HyperParameterScheduler('entropy_beta', [(40, 0.005), (80, 0.001)]))

    return TrainConfig(
        dataset=dataflow,
        optimizer=optimizer,
        callbacks = Callbacks(callbacks),
        extra_threads_procs=[master],
        session_config=session_config,
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
        max_epoch=args.max_epoch,
        extra_arg=extra_arg
    )


if __name__ == '__main__':
    import os

    os.setpgrp()

    tf.logging.set_verbosity(tf.logging.ERROR)
    import logging

    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    print "args.mkl == ", args.mkl
    ENV_NAME = args.env
    p = get_player();
    del p  # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.mkl:
        print "using MKL convolution"
        label_map = {"Conv2D": "MKL",
                     "Conv2DBackpropFilter": "MKL",
                     "Conv2DBackpropInput": "MKL"}
    else:
        print "using tensorflow convolution"
        label_map = {}
    with ops.Graph().as_default() as g:
        tf.set_random_seed(my_task_index)
        np.random.seed(my_task_index)
        with g._kernel_label_map(label_map):
            with tf.device('/job:worker/task:{}/cpu:0'.format(my_task_index)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                    if args.task != 'train':
                        cfg = PredictConfig(
                            model=Model('/job:ps/task:0/cpu:0'),
                            session_init=SaverRestore(args.load),
                            input_var_names=['state'],
                            output_var_names=['logits:0'])
                        if args.task == 'play':
                            play_model(cfg)
                        elif args.task == 'eval':
                            eval_model_multithread(cfg, EVAL_EPISODE)
                    else:
                        nr_towers = args.nr_towers
                        predict_towers = args.nr_predict_towers * [0, ]

                        if args.cpu != 1:
                            nr_gpu = get_nr_gpu()
                            if nr_gpu > 1:
                                predict_tower = range(nr_gpu)[-nr_gpu / 2:]
                            else:
                                predict_tower = [0]

                        chief_worker_hostname = _chief_worker_hostname(cluster)
                        config = get_config(args, is_chief, my_task_index, chief_worker_hostname, len(cluster['worker']))
                        if args.load:
                            config.session_init = SaverRestore(args.load)
                        config.tower = range(nr_towers)

                        logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
                            ','.join(map(str, config.tower)), ','.join(map(str, predict_towers))))

                        if args.sync:
                            logger.info('using sync version')
                            SyncMultiGPUTrainer(config, predict_tower=predict_towers).train()
                        else:
                            logger.info('using async version')
                            while True:
                                try:
                                    trainer = AsyncMultiGPUTrainer(config, predict_tower=predict_towers)
                                    trainer.train()
                                except Exception as e:
                                    print ('===== EXCEPTION IN TRAIN-ATARI.PY [{}] ======'.format(os.environ['SLURMD_NODENAME']))
                                    traceback.print_exc()
