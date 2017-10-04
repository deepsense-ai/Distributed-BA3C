import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

from tensorpack.RL.common import MapPlayerState
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.RL import *

from tensorflow_slurm_utils import tf_server_from_slurm
import neptune_mp_server
import time
from six.moves import queue
from threading import Thread

import argparse
import cv2
import json

ENV_NAME="Breakout-v0"

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

## ugly hack:
TARGET_CHANNELS = 16
#IMAGE_SHAPE3 = IMAGE_SIZE + (TARGET_CHANNELS,)
EVAL_EPISODE = 50

parser = argparse.ArgumentParser()
parser.add_argument('--fc_neurons', required=True, type=int)
parser.add_argument('--fc_splits', required=True, type=int)
parser.add_argument('--replace_with_conv', required=True, type=bool)
parser.add_argument('--env', required=False, default='Breakout-v0', type=str)
parser.add_argument('--models_dir', required=True, type=str)
parser.add_argument('--server_port', required=True, type=int)
parser.add_argument('--ps', required=True, type=int)

args = parser.parse_args()

cluster, my_job_name, my_task_index = tf_server_from_slurm(ps_number=args.ps, port_number=0)
args.server_host = cluster['worker'][0].split(':')[0]

def get_player(viz=False, train=False, dumpdir=None, videofile=None):
    pl = GymEnv(args.env, dumpdir=dumpdir, record=videofile)

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

p = get_player();
del p  # set NUM_ACTIONS

def Conv2D(name, x, out_channel, kernel_shape, padding, stride=1, nl=tf.nn.relu):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[-1]
    kernel_shape = [kernel_shape, kernel_shape]
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel, out_channel]
    stride = [1, stride, stride, 1]

    W = tf.get_variable(name+'/W', filter_shape)

    x = tf.to_float(x)
    W = tf.to_float(W)
    conv = tf.nn.conv2d(x, W, stride, padding)
    return nl(conv, name='output')

def MaxPooling(name, x, shape, stride=None, padding='VALID'):
    padding = padding.upper()
    shape = [1, shape, shape, 1]
    if stride is None:
        stride = shape
    else:
        stride = [1, stride, stride, 1]
    x = tf.to_float(x)

    return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)

def batch_flatten(x):
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, np.prod(shape)])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

def FullyConnected(name, x, out_dim, use_bias=True, nl=tf.nn.relu):
    x = batch_flatten(x)
    in_dim = x.get_shape().as_list()[1]

    W = tf.get_variable(name+'/W', [in_dim, out_dim])
    if use_bias:
        b = tf.get_variable(name+'/b', [out_dim])

    prod = tf.nn.xw_plus_b(x, W, b) if use_bias else tf.matmul(x, W)
    return nl(prod, name='output')

def get_NN_prediction(image):
    with tf.device('/cpu:0'):
        image = image / 255.0

        print(CHANNEL)
        dummy_channels = TARGET_CHANNELS - CHANNEL
        image = tf.concat([image, tf.zeros((tf.shape(image)[0], 84, 84, dummy_channels))], 3)
        input_sum = tf.reduce_sum(tf.abs(image))
        l = Conv2D('conv0', image, out_channel=32, kernel_shape=5, padding='VALID')
        l = MaxPooling('pool0', l, 2)

        l = Conv2D('conv1', l, out_channel=32, kernel_shape=5, padding='VALID')
        l = MaxPooling('pool1', l, 2)

        l = Conv2D('conv2', l, out_channel=64, kernel_shape=5, padding='VALID')
        l = MaxPooling('pool2', l, 2)

        l = Conv2D('conv3', l, out_channel=64, kernel_shape=3, padding='VALID')

        # 1 fully connected layer but split into many tensors
        # to better split parameters between many PS's
        if args.replace_with_conv:
            fc_splits = []
            neurons = args.fc_neurons / args.fc_splits
            for i in range(args.fc_splits):
                fc = Conv2D('fc1_{}'.format(i), l, out_channel=neurons, kernel_shape=5,
                             nl=tf.identity, padding='VALID')

                fc = tf.reshape(tensor=fc, shape=[-1, neurons])
                fc_splits.append(fc)
            l = tf.concat(fc_splits, axis=1)
        else:
            fc_splits = []
            neurons = args.fc_neurons / args.fc_splits
            for i in range(args.fc_splits):
                fc = FullyConnected('fc1_{}'.format(i), l, out_channel=neurons, nl=tf.identity)

                fc = tf.nn.relu(fc_part, 'relu1')
                fc_splits.append(fc)
            l = tf.concat(fc_splits, axis=1)


        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        value = FullyConnected('fc-v', l, out_dim=1, nl=tf.identity)

        return policy, value

def build_graph(state):
    with tf.device('/cpu:0'):
        logits, value = get_NN_prediction(state)
        policy = tf.nn.softmax(logits, name='logits')
        return policy

def eval_model(model_path, worker_num=50):
    def worker(worker_id, output_queue, input_queue, score_queue):
        player = get_player()
        spc = player.get_action_space()
        score = 0.
        isOver = False

        while not isOver:
            image = player.current_state()
            output_queue.put((image, worker_id))
            dist = input_queue.get()
            a = np.argmax(dist)
            if np.random.random() < 0.001:
                act = spc.sample()

            r, isOver = player.action(a)
            score += r

        output_queue.put((None, worker_id))
        score_queue.put(score)

    output_queue = queue.Queue()
    input_queues = [queue.Queue() for _ in range(worker_num)]
    score_queue = queue.Queue()

    threads = [Thread(target=worker, args=(i, output_queue, q, score_queue))
                for i, q in zip(range(worker_num), input_queues)]
    for t in threads:
        t.start()
        time.sleep(0.1)

    alive = worker_num
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        while alive > 0:
            images = []
            ids = []
            for i in range(alive):
                image, worker_id = output_queue.get()
                if image is None:
                    alive -= 1
                else:
                    images.append(image)
                    ids.append(worker_id)

            if alive == 0:
                break

            p = sess.run(eval_model.policy, feed_dict={eval_model.state: images})
            for worker_id, dist in zip(ids, p):
                input_queues[worker_id].put(dist)

    for t in threads:
        t.join()

    scores = []
    for _ in range(worker_num):
        scores.append(score_queue.get())
    return np.mean(scores), np.max(scores)

eval_model.state = tf.placeholder(tf.float32, shape=(None,)+IMAGE_SHAPE3)
eval_model.policy = build_graph(eval_model.state)


if __name__ == '__main__':
    neptune_client = neptune_mp_server.Client(server_host=args.server_host, server_port=args.server_port)

    i = 0
    dir_path = os.path.join(args.models_dir, 'iter_{}')
    # print("WAITING FOR DIR {}".format(dir_path.format(i)))
    while True:
        if os.path.isdir(dir_path.format(i)):
            dir_path_i = dir_path.format(i)
            time.sleep(0.1)
            for f in os.listdir(dir_path_i):
                if f[:5] == 'model':
                    model_path = os.path.join(dir_path_i, f)
                    break
            model_time = float(model_path.split('-')[1])
            model_steps = float(model_path.split('-')[2])
            model_mean, model_max = eval_model(model_path, EVAL_EPISODE)

            content = ('score', model_mean, model_max, model_time)
            message = (0, content)
            neptune_client.send(message)

            i += 1
            while os.path.isdir(dir_path.format(i+1)):
                i += 1

            info = {
                    'mean_score' : model_mean,
                    'max_score' : model_max,
                    'fc_neurons' : args.fc_neurons,
                    'fc_splits' : args.fc_splits,
                    'replace_with_conv' : args.replace_with_conv,
                    'global_step' : model_steps,
                    'time' : model_time
                    }
            with open(os.path.join(dir_path_i, 'info.json'), 'w') as f:
                json.dump(info, f)

            print("WAITING FOR DIR {}".format(dir_path.format(i)))

        time.sleep(3)

