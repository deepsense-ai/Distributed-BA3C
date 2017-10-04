import os
import tensorflow as tf
import sys

sys.path.append(sys.argv[1])
from tensorflow_slurm_utils import tf_server_from_slurm
c, n, i = tf_server_from_slurm(ps_number=int(sys.argv[2]))
t = 0
if sys.argv[3] == 'True':
    t += 1
if sys.argv[4] == 'True':
    t += 1

if n == 'worker':
    if t == 2:
        if sys.argv[3] == 'True' and i == len(c['worker']) - 2:
                print 'eval'
                sys.exit()
        if sys.argv[4] == 'True' and i == len(c['worker']) - 1:
                print 'record'
                sys.exit()
    elif t == 1:
        if sys.argv[3] == 'True' and i == len(c['worker']) - 1:
                print 'eval'
                sys.exit()
        if sys.argv[4] == 'True' and i == len(c['worker']) - 1:
                print 'record'
                sys.exit()
    print i
else:
    print 'ps'

