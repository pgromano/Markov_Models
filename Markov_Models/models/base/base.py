

class BaseModel(object):
    def __init__(self, data, rev=True, sparse=False, **kwargs):
        self._is_sparse = sparse
        self._is_reversible = rev
        self.data = data
        self.n_sets = len(data)
        self.n_samples = [self.data[i].shape[0] for i in range(self.n_sets)]
        self.n_features = [self.data[i].shape[1] for i in range(self.n_sets)]
