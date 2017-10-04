import os

class Neptune(object):
    def Context():
        return NeptuneContextWrapper()

    @property
    def ChannelType(self):
        return NeptuneChannelTypes()

class NeptuneChannelTypes(object):
    @property
    def NUMERIC(self):
        return float


neptune = Neptune()

class NeptuneContextWrapper(object):
    def __init__(self, experiment_dir):
        self._job = JobWrapper(experiment_dir, to_file=True)

    @property
    def job(self):
        return self._job

class JobWrapper(object):
    def __init__(self, experiment_dir, to_file=True):
        self.to_file = to_file
        self.experiment_dir = experiment_dir

    def create_channel(self, *args, **kwargs):
        return ChannelWrapper(kwargs['name'], self.experiment_dir, to_file=self.to_file)

    def create_chart(self, *args, **kwargs):
        pass

class ChannelWrapper(object):
    def __init__(self, name, experiment_dir, to_file=True):
        self.to_file = to_file
        self.name = name
        if self.to_file:
            self.filename = os.path.join(experiment_dir, self.name + '.csv')
            self.fd = open(self.filename, 'w')
            header = '{},{}\n'.format('x', 'y')
            self.fd.write(header)
            self.fd.flush()

    def send(self, x, y, *args, **kwargs):
        print "##### Sending to neptune: ", self.name,": ", x,",", y, "#####"
        if self.to_file:
            self._send_to_file(x,y)

    def _send_to_file(self, x, y):
        line = '{},{}\n'.format(x, y)
        self.fd.write(line)
        self.fd.flush()

