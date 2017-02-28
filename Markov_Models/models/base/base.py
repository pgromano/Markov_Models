import numpy as np

class BaseModel(object):
    def __init__(self, data, rev=True, sparse=False, **kwargs):
        self._is_sparse = sparse
        self._is_reversible = rev
        self.data = data
        self.n_sets = len(data)
        self.n_samples = [self.data[i].shape[0] for i in range(self.n_sets)]
        n_features = [self.data[i].shape[1] for i in range(self.n_sets)]
        if np.all(np.equal(n_features, n_features[0])):
            self.n_features = n_features[0]
        else:
            raise AttributeError('''
            Number of features must be the same for all sets of data!''')
