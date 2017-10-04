#!/usr/bin/env python2

import sys
import os
import argparse
import random
import sys
import time
# ugly hack
sys.path.append('./tensorpack_cpu/examples/OpenAIGym')


parser = argparse.ArgumentParser()
parser.add_argument('--njobs', '-n', help='Number of nodes', required=True, type=int)
parser.add_argument('--ngrads', '-g', help='Number of gradients for SyncReplicasOptimizer', type=int)
parser.add_argument('--cores', '-c', help='', required=True, type=int)
parser.add_argument('--optimizer', '-o', help='Optimizer to use', default='adam', type=str,
        choices=['adam', 'gd', 'adagrad', 'adadelta', 'momentum', 'rms'])
parser.add_argument('--environment', '-e', help='', default='Breakout-v0', type=str)
parser.add_argument('--use_sync', help='Set to use SyncReplicasOptimizer', action='store_true')
parser.add_argument('--tags', '-t', help='Set tags for neptune experiment', action='append', type=str)
parser.add_argument('--name', help='Set name for neptune experiment', default='test_neptunized', type=str)
parser.add_argument('--lr', '-l', help='Learning rate', default=0.00015, type=float)
parser.add_argument('--batch_size', '-b', help='Batch size', default=128, type=int)
parser.add_argument('--intel_tf', help='Set to use TF from Intel', action='store_true')
parser.add_argument('--fc_neurons', help='Number of neurons in the FC layer', default=256, type=int)
parser.add_argument('--simulator_procs', help='Number of simulator processes per node', default=100, type=int)
parser.add_argument('--early_stopping', '-s', help='Set early stopping for game, set 0 for default for this game', type=float)
parser.add_argument('--ps', help='Number of parameter servers to use', required=False, default=1, type=int)
parser.add_argument('--fc_init', help='Initialization of the fully connected layer', required=False, default='uniform', type=str, choices=['normal', 'uniform'])
parser.add_argument('--conv_init', help='Initialization of the convolution layer', required=False, default='normal', type=str, choices=['normal', 'uniform', 'xavier'])
parser.add_argument('--offline', help='dont use web neptune', action='store_true')
parser.add_argument('--exp_dir', help='experiment directory for offline experiments', default=".", type=str)
parser.add_argument('--use_normal_fc', required=False, action='store_true')
parser.add_argument('--fc_splits', required=False, default=1, type=int)
parser.add_argument('--debug_charts', required=False, action='store_true', help='set to show debug charts in neptune')
parser.add_argument('--log_dir', required=False, default='/net/archive/groups/plggluna/intel_2/logs/', type=str)
parser.add_argument('--short', required=False, action='store_true')
parser.add_argument('--epsilon', required=False, default=1e-8, type=float, help='Epsilon in adam optimizer')
parser.add_argument('--beta1', required=False, default=0.9, type=float, help='Beta1 parameter in Adam')
parser.add_argument('--beta2', required=False, default=0.999, type=float, help='Beta2 parameter in Adam')
parser.add_argument('--save_every', required=False, default=0, type=int)
parser.add_argument('--adam_debug', required=False, action='store_true')
parser.add_argument('--save_output', required=False, action='store_true')
parser.add_argument('--eval_node', required=False, action='store_true')
parser.add_argument('--record_node', required=False, action='store_true')
parser.add_argument('--record_length', required=False, default=120, type=str)
parser.add_argument('--schedule_hyper', required=False, action='store_true')

timestamp = time.time()
args=parser.parse_args()
args.name = args.name + "_" + str(timestamp)

print "args.offline: ", args.offline

if args.use_sync and args.ngrads is None:
    parser.error('if using SyncReplicasOptimizer you have to specify --ngrads argument')
    sys.exit(1)

if not args.use_sync:
    args.ngrads = 1

port=random.randint(1025, 2 ** 16 - 2)
tf_port=random.randint(1025, 2 ** 16 - 2)

while port == tf_port:
    tf_port=random.randint(1025, 2 ** 16 - 2)

if args.tags is None:
    args.tags = []

#add tags for number of nodes and cores
if args.njobs == 1:
    args.tags.append("1node")
else:
    args.tags.append("{}nodes".format(args.njobs - 2))

if args.cores == 1:
    args.tags.append("1core")
else:
    args.tags.append("{}cores".format(args.cores))

if args.early_stopping is not None:
    if args.early_stopping == 0:
        if args.environment == 'Breakout-v0':
            args.early_stopping = 300.
        elif args.environment == 'Pong-v0':
            args.early_stopping = 10.
        elif args.environment == 'Riverraid-v0':
            args.early_stopping = 4000.
        elif args.environment == 'Seaquest-v0':
            args.early_stopping = 1830.
        elif args.environment == 'SpaceInvaders-v0':
            args.early_stopping = 700.
else:
    args.early_stopping = 'None'

if args.short:
    time_limit = '55:00'
else:
    time_limit = '6:00:00'

if args.log_dir[-1] != '/':
    args.log_dir += '/'

if args.eval_node or args.record_node:
    assert args.save_every > 0

bash_command = "srun -A luna -N {njobs} -n {njobs} -c {c} -t {time_limit} distributed_tensorpack_mkl.sh {port} {tf_port} {env} {opt} {use_sync} \"{tags}\" \"{name}\" {lr} {batch_size} {ngrads} {intel_tf} {early} {fc_neurons} {simulator_procs} {ps} {fc_init} {conv_init} {offline} {exp_dir} {replace_with_conv} {fc_splits} {debug_charts} {log_dir} {epsilon} {beta1} {beta2} {save} {adam_debug} {save_output} {eval_node} {record_node} {record_length} {schedule_hyper}".format(
        njobs=args.njobs,
        c=args.cores,
        port=port,
        tf_port=tf_port,
        env=args.environment,
        opt=args.optimizer,
        use_sync=(1 if args.use_sync else 0),
        tags=' '.join(args.tags),
        name=args.name,
        lr=args.lr,
        batch_size=args.batch_size,
        ngrads=args.ngrads,
        intel_tf=(1 if args.intel_tf else 0),
        ps=args.ps,
        fc_neurons=args.fc_neurons,
        simulator_procs=args.simulator_procs,
        early=args.early_stopping,
        fc_init=args.fc_init,
        conv_init=args.conv_init,
        offline=str(args.offline),
        exp_dir=args.exp_dir,
        replace_with_conv=(not args.use_normal_fc),
        fc_splits=args.fc_splits,
        time_limit=time_limit,
        debug_charts=args.debug_charts,
        log_dir=args.log_dir,
        epsilon=args.epsilon,
        beta1=args.beta1,
        beta2=args.beta2,
        save=args.save_every,
        adam_debug=args.adam_debug,
        save_output=args.save_output,
        eval_node=args.eval_node,
        record_node=args.record_node,
        record_length=args.record_length,
        schedule_hyper=args.schedule_hyper
        )
print("bash command: ", bash_command)
os.system(bash_command)
