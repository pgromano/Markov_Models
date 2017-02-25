import Markov_Models as mm

class MSM(object):
    def __init__(self, data, rev=True, sparse=False):
        self._is_sparse = sparse
        self._is_reversible = rev
        self.data = data
        self.n_sets = len(data)
        self.n_samples = [self.data[i].shape[0] for i in range(self.n_sets)]
        self.n_features = [self.data[i].shape[1] for i in range(self.n_sets)]

        self.microstates = mm.models.base.BaseMicroMSM(self)
        self.macrostates = mm.models.base.BaseMacroMSM(self)


class HMM(object):
    def __init__(self, data, **kwargs):
        self.data = data
        self.n_sets = len(data)
        self.n_samples = [self.data[i].shape[0] for i in range(self.n_sets)]
        self.n_features = [self.data[i].shape[1] for i in range(self.n_sets)]

class Markov_Chain(object):
    def __init__(self, data, **kwargs):
        self.data = data
        self.n_sets = len(data)
        self.n_samples = [self.data[i].shape[0] for i in range(self.n_sets)]
        self.n_features = [self.data[i].shape[1] for i in range(self.n_sets)]

from Markov_Models import util
class Load(object):
    def __init__(self, files):
        self.files = files

    @property
    def from_CSV(self):
        return util.load.from_CSV(self.files)

    @property
    def from_ASCII(self):
        return util.load.from_ASCII(self.files)
