import unittest

class TfDictOp(object):
    def __init__(self, ops_dict):
        '''
        ops_dict should be dict from strings to tensorflow ops
        '''
        self.indices = [name for name in ops_dict]
        self.ops = [ops_dict[name] for name in self.indices]

    '''
    creates list of tensorflow operations for sess.run
    '''
    @property
    def op(self):
        return self.ops

    '''
    creates dict from sess.run output
    '''
    def get_output(self, tf_output):
        return {
                name : output for name, output in zip(self.indices, tf_output)
                }

class BasicTest(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)

        self.input = {
                'a' : 1,
                'b' : 2,
                'c' : 3
                }

        self.tf_dict = TfDictOp(self.input)

    def test_op(self):
        ops = self.tf_dict.op

        self.assertIn(1, ops)
        self.assertIn(2, ops)
        self.assertIn(3, ops)

    def test_get_output(self):
        ops = self.tf_dict.op
        outputs = self.tf_dict.get_output([10, 20, 30])

        def get_name(val):
            for name in self.input:
                if self.input[name] == val:
                    return name

        self.assertEqual(10, outputs[get_name(ops[0])])
        self.assertEqual(20, outputs[get_name(ops[1])])
        self.assertEqual(30, outputs[get_name(ops[2])])

if __name__ == '__main__':
    unittest.main()
