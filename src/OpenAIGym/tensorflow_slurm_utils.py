from __future__ import print_function
from random import randint
import unittest
import re
import os

def tf_server_from_slurm(ps_number, port_number=2222):
    """
    Creates a tensorflow.train.Server from environment variables
    provided by the slurm cluster management system. The server
    is not started.

    @param: ps_number number of parameter servers to run
    @param: port_number port number to be used for communication
    @return: a named tuple containing cluster with fields cluster_spec,
             task_name and task_id
    """

    nodelist = os.environ["SLURM_JOB_NODELIST"]
    nodename = os.environ["SLURMD_NODENAME"]
    nodelist = _expand_nodelist(nodelist)
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))

    if len(nodelist) != num_nodes:
        raise ValueError("Number of slurm nodes {} not equal to {}".format(len(nodelist), num_nodes))

    if nodename not in nodelist:
        raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(nodename,nodelist))

    ps_nodes = [node for i, node in enumerate(nodelist) if i < ps_number]
    worker_nodes = [node for i, node in enumerate(nodelist) if i >= ps_number]

    if nodename in ps_nodes:
        my_job_name = "ps"
        my_task_index = ps_nodes.index(nodename)
    else:
        my_job_name = "worker"
        my_task_index = worker_nodes.index(nodename)

    worker_sockets = [":".join([node, str(port_number + 1)]) for node in worker_nodes]
    ps_sockets = [":".join([node, str(port_number)]) for node in ps_nodes]
    cluster = {"worker": worker_sockets, "ps" : ps_sockets}

    return cluster, my_job_name, my_task_index

def _pad_zeros(iterable, length):
    return (str(t).rjust(length, '0') for t in iterable)

def _expand_ids(ids):
    ids = ids.split(',')
    result = []
    for id in ids:
        if '-' in id:
            begin, end = [int(token) for token in id.split('-')]
            result.extend(_pad_zeros(range(begin, end+1), len(token)))
        else:
            result.append(id)
    return result

def _expand_nodelist(nodelist):
    prefix, ids = re.findall("(.*)\[(.*)\]", nodelist)[0]
    ids = _expand_ids(ids)
    result = [prefix + str(id) for id in ids]
    return result

def _worker_task_id(nodelist, nodename):
    return nodelist.index(nodename)

#TODO: test the case with p0912, p0913
#TODO: test the case with p0912-p0920

class BasicTestData(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.nodelist = 'p[1135,1137-1142,1147-1148,1152]'
        self.first_nodename = 'p1135'
        self.nodename = 'p1140'
        self.nodes_number = 10

class ShortNodenameTestData(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.nodelist = 'p[0900-0910]'
        self.first_nodename = 'p0900'
        self.nodename = 'p0902'
        self.nodes_number = 11

class ShortNodenameTestData2(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.nodelist = 'p0900,p0910]'
        self.first_nodename = 'p0900'
        self.nodename = 'p0910'
        self.nodes_number = 2

class TensorflowSlurmUtilsTest(object):
    def test_expand_ids(self):
        test_ids = '1-5,7,8-12'
        res = _expand_ids(test_ids)
        print(res)

    def test_expand_nodelist(self):
        expanded = _expand_nodelist(self.nodelist)
        print(expanded)
        self.assertEqual(len(expanded), self.nodes_number)
        self.assertIn(self.nodename, expanded)

    def test_first_task_id(self):
        expanded = _expand_nodelist(self.nodelist)
        first_task_id = _worker_task_id(expanded, self.first_nodename)
        self.assertEqual(first_task_id, 0)

    def test_other_task_id(self):
        expanded = _expand_nodelist(self.nodelist)
        task_id = _worker_task_id(expanded, self.nodename)
        self.assertIn(task_id, range(self.nodes_number))

    def test_tf_server_from_slurm(self):
        os.environ["SLURM_JOB_NODELIST"] = self.nodelist
        os.environ["SLURMD_NODENAME"] = self.nodename
        os.environ["SLURM_JOB_NUM_NODES"] = str(self.nodes_number)
        server = tf_server_from_slurm(ps_number=2)
        print(server)

class BasicTestCase(BasicTestData, TensorflowSlurmUtilsTest):
    pass
class ShortNodenameTestCase(ShortNodenameTestData, TensorflowSlurmUtilsTest):
    pass
class ShortNodenameTestCase2(ShortNodenameTestData2, TensorflowSlurmUtilsTest):
    pass

if __name__ == '__main__':
    unittest.main()
