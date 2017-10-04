import zmq
import json
import StringIO
from time import time, sleep
import numpy as np
import os
from tensorpack.utils.neptune_utils import Neptune, NeptuneContextWrapper, JobWrapper, ChannelWrapper
import traceback

neptune = Neptune()

class Server(object):
    def __init__(self, number_of_workers, port, debug_charts, adam_debug, schedule_hyper, experiment_dir):
        self.port = port
        self.debug_charts = debug_charts
        self.adam_debug = adam_debug
        self.schedule_hyper = schedule_hyper

        self.neptune_ctx = NeptuneContextWrapper(experiment_dir)
        job = self.neptune_ctx.job

        self.workers_set = set()

        def create_channel(name, channel_type=neptune.ChannelType.NUMERIC):
            return job.create_channel(name=name, channel_type=channel_type)

        def create_per_worker_channels_and_chart(name):
            channels = [create_channel('{}_{}'.format(name, i)) for i in range(number_of_workers)]
            job.create_chart(
                name=name,
                series={
                    '{}_{}'.format(name, i) : channel for i, channel in enumerate(channels)
                    })
            return channels

        self.score_mean_channel = create_channel('score_mean')
        self.score_max_channel = create_channel('score_max')
        self.online_score_channel = create_channel('online_score')
        job.create_chart(
            name='score',
            series={
                'score_mean' : self.score_mean_channel,
                'score_max' : self.score_max_channel,
                'online_score' : self.online_score_channel
                })

        self.cost_channel = create_channel('cost')
        self.policy_loss_channel = create_channel('policy_loss')
        self.xentropy_loss_channel = create_channel('xentropy_loss')
        self.max_logit_channel = create_channel('max_logit')
        self.value_loss_channel = create_channel('value_loss')
        self.advantage_channel = create_channel('advantage')
        self.pred_reward_channel = create_channel('pred_reward')
        job.create_chart(
            name='loss',
            series= {
                'cost' : self.cost_channel,
                'policy_loss' : self.policy_loss_channel,
                'xentropy_loss' : self.xentropy_loss_channel,
                'value_loss' : self.value_loss_channel,
                'advantage' : self.advantage_channel,
                'pred_reward' : self.pred_reward_channel,
                'max_logit' : self.max_logit_channel
                })

        self.active_workes_channel = create_channel('active_workers')
        self.dp_per_s_channel = create_channel('dp_per_s')
        job.create_chart(
            name='other',
            series={
                'active_workers' : self.active_workes_channel,
                'datapoints/s' : self.dp_per_s_channel
                })

        self.active_relus_channel = create_channel('active_relus')
        job.create_chart(
            name='active relus',
            series={
                'active_relus' : self.active_relus_channel
                })

        self.max_delay_channel = create_channel('max_delay')
        self.mean_delay_channel = create_channel('mean_delay')
        self.min_delay_channel = create_channel('min_delay')
        job.create_chart(
            name='delay',
            series={
                'max_delay' : self.max_delay_channel,
                'mean_delay' : self.mean_delay_channel,
                'min_delay' : self.min_delay_channel
                })

        if self.adam_debug:
            self.adam_m_norm_channel = create_channel('adam_m_norm')
            self.adam_v_norm_channel = create_channel('adam_v_norm')
            self.adam_update_norm_channel = create_channel('adam_update_norm')
            self.adam_lr_channel = create_channel('adam_lr')
            job.create_chart(
                name='adam_state',
                series={
                    'm_norm' : self.adam_m_norm_channel,
                    'v_norm' : self.adam_v_norm_channel,
                    'update_norm' : self.adam_update_norm_channel,
                    'lr' : self.adam_lr_channel
                    })

        if self.schedule_hyper:
            self.learning_rate_channel = create_channel('learning_rate')
            self.entropy_beta_channel = create_channel('entropy_beta')
            job.create_chart(
                name='scheduled hyperparams',
                series={
                    'learning rate' : self.learning_rate_channel,
                    'entropy beta' : self.entropy_beta_channel
                    })

        if not self.debug_charts:
            self.start_time = time()
            return

        self.grad_norm_before_clip_channel = create_channel('grad_norm_before_clip')
        self.grad_norm_after_clip_channel = create_channel('grad_norm_after_clip')
        job.create_chart(
            name='gradients',
            series={
                'grad_norm_before_clip' : self.grad_norm_before_clip_channel,
                'grad_norm_after_clip' : self.grad_norm_after_clip_channel
                })

        self.cost_channels = create_per_worker_channels_and_chart('cost')
        self.xentropy_loss_channels = create_per_worker_channels_and_chart('xentropy_loss')
        self.value_loss_channels = create_per_worker_channels_and_chart('value_loss')
        self.policy_loss_channels = create_per_worker_channels_and_chart('policy_loss')

        self.mean_value_channels = create_per_worker_channels_and_chart('mean_value')
        self.mean_state_channels = create_per_worker_channels_and_chart('mean_state')
        self.mean_action_channels = create_per_worker_channels_and_chart('mean_action')
        self.mean_future_reward_channels = create_per_worker_channels_and_chart('mean_futurereward')
        self.mean_init_R_channels = create_per_worker_channels_and_chart('mean_init_R')
        self.games_over_channels = create_per_worker_channels_and_chart('games_over')

        self.fc_value_channels = create_per_worker_channels_and_chart('fc_value')
        self.fc_fc0_channels = create_per_worker_channels_and_chart('fc_fc0')

        self.conv0_out_channels = create_per_worker_channels_and_chart('conv0_out')
        self.conv1_out_channels = create_per_worker_channels_and_chart('conv1_out')
        self.conv2_out_channels = create_per_worker_channels_and_chart('conv2_out')
        self.conv3_out_channels = create_per_worker_channels_and_chart('conv3_out')
        self.fc1_0_out_channels = create_per_worker_channels_and_chart('fc1_0_out')
        self.fc_pi_out_channels = create_per_worker_channels_and_chart('fc_pi_out')
        self.fc_v_out_channels = create_per_worker_channels_and_chart('fc_v_out')

        self.start_time = time()

    def _get_hours_since_start(self):
        return (time() - self.start_time) / (60. * 60.)

    def _get_minutes_since(self, t):
        return (time() - t) / 60.

    def _send_per_worker_loss(self, x, id, content):
        # original x may not be strictly increasing
        x = self._get_hours_since_start()
        self.cost_channels[id].send(x, content[1])
        self.policy_loss_channels[id].send(x, content[2])
        self.xentropy_loss_channels[id].send(x, content[3])
        self.value_loss_channels[id].send(x, content[4])

    def _dump_to_channels(self, id, content):
        x = self._get_hours_since_start()
        self.workers_set.add(id) # add this worker to active workers
        if content[0] == 'score':
            if len(content) == 4:
                x = content[3]
            self.score_mean_channel.send(x, content[1])
            self.score_max_channel.send(x, content[2])
        elif content[0] == 'loss':
            self.cost_channel.send(x, content[1])
            self.policy_loss_channel.send(x, content[2])
            self.xentropy_loss_channel.send(x, content[3])
            self.value_loss_channel.send(x, content[4])
            self.advantage_channel.send(x, content[5])
            self.pred_reward_channel.send(x, content[6])
            self.max_logit_channel.send(x, content[7])

            if self.debug_charts:
                self._send_per_worker_loss(x, id, content)
        elif content[0] == 'online':
            self.online_score_channel.send(x, content[1])
        elif content[0] == 'other':
            self.active_relus_channel.send(x, content[1])
            self.dp_per_s_channel.send(x, content[2])
        elif content[0] == 'delays':
            self.mean_delay_channel.send(x, content[1][0])
            self.max_delay_channel.send(x, content[1][1])
            self.min_delay_channel.send(x, content[1][2])

        if self.adam_debug and content[0] == 'adam':
            self.adam_m_norm_channel.send(x, content[1])
            self.adam_v_norm_channel.send(x, content[2])
            self.adam_update_norm_channel.send(x, content[3])
            self.adam_lr_channel.send(x, content[4])

        if self.schedule_hyper and content[0] == 'schedule':
            self.learning_rate_channel.send(x, content[1])
            self.entropy_beta_channel.send(x, content[2])

        if not self.debug_charts:
            return

        if content[0] == 'grad':
            self.grad_norm_before_clip_channel.send(x, content[1])
            self.grad_norm_after_clip_channel.send(x, content[2])
        elif content[0] == 'other':
            self.fc_fc0_channels[id].send(content[4], content[3])
            self.fc_value_channels[id].send(content[4], content[5])
        elif content[0] == 'env_state':
            x = self._get_hours_since_start()
            self.mean_state_channels[id].send(x, content[1])
            self.mean_future_reward_channels[id].send(x, content[2])
            self.mean_value_channels[id].send(x, content[3])
            self.mean_init_R_channels[id].send(x, content[4])
            self.mean_action_channels[id].send(x, content[5])
            self.games_over_channels[id].send(x, content[6])
            #self.mean_state_channel.send(x, content[1])
        elif content[0] == 'layers':
            self.conv0_out_channels[id].send(x, content[1])
            self.conv1_out_channels[id].send(x, content[2])
            self.conv2_out_channels[id].send(x, content[3])
            self.conv3_out_channels[id].send(x, content[4])
            self.fc1_0_out_channels[id].send(x, content[5])
            self.fc_pi_out_channels[id].send(x, content[6])
            self.fc_v_out_channels[id].send(x, content[7])

    def main_loop(self):
        print 'server main loop'
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        hostname = os.environ['SLURMD_NODENAME']
        address = "tcp://*:{}".format(self.port)
        print 'before socket bind... {}'.format(address)
        socket.bind(address)

        last_workers_check = time()
        while True:
            try:
                print 'receiving'
                message = socket.recv()
                # just a trash message to reset the socket
                socket.send('ACK')
                id, content = json.loads(message)
                self._dump_to_channels(id, content)
                print content

                # every 1 minutes count number of workers that server got any message from
                # in last 1 minutes and sends it to neptune
                if self._get_minutes_since(last_workers_check) > 1:
                    x = self._get_hours_since_start()
                    self.active_workes_channel.send(x, len(self.workers_set))
                    self.workers_set.clear()
                    last_workers_check = time()

            except Exception as e:
                print '======= EXCEPTION IN MP SERVER MAIN LOOP ======='
                print e.message
                traceback.print_exc()
                break
        socket.close()

class Client(object):
    def __init__(self, server_host, server_port):
        self.server_port = server_port
        self.server_host = server_host

    def send(self, message):
        sio = StringIO.StringIO()
        json.dump(message, sio)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        address = "tcp://{host}:{port}".format(host=self.server_host, port=self.server_port)
        print "sending to address {}".format(address)
        socket.connect(address)
        socket.send(sio.getvalue())
        socket.close()

if __name__ == '__main__':
    number_of_workers = 5
    import multiprocessing as mp
    def dummy_client(id):
        np.random.seed(id)
        c = Client(server_host='localhost')
        v = [0,0]
        while True:
            print 'sending...'
            v[0] += np.random.random()
            v[1] += np.random.random() * 2
            message = (id, v)
            c.send(message)
            sleep(1)
    for i in range(number_of_workers):
        dummy = mp.Process(target=dummy_client, args=[i])
        dummy.start()
    server = Server(number_of_workers=number_of_workers)
    server.main_loop()


