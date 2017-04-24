import numpy as np

class BaseModel(object):
    def __init__(self, data, **kwargs):
        self._is_force_db = kwargs.get('db', False)
        self._is_reversible = kwargs.get('rev', True)
        self._is_sparse = kwargs.get('sparse', False)

        data = _check_data_structure(data)
        self.data = data
        self.n_sets = len(data)
        self.n_samples = [self.data[i].shape[0] for i in range(self.n_sets)]

        n_features = [self.data[i].shape[1] for i in range(self.n_sets)]
        if np.all(np.equal(n_features, n_features[0])):
            self.n_features = n_features[0]
        else:
            raise AttributeError('''
            Number of features must be the same for all sets of data!''')

    def histogram(self, axis=None, bins=100):
        if axis is None:
            his, ext = np.histogramdd(np.concatenate([self.data[i] for i in range(self.n_sets)]), bins=bins)
            extent = []
            for k in range(self.n_features):
                extent.append(ext[k].min())
                extent.append(ext[k].max())
            return his, extent
        else:
            his, ext = np.histogramdd(np.concatenate([self.data[i][:,axis] for i in range(self.n_sets)]), bins=bins)
            extent = []
            for k in range(len(ext)):
                extent.append(ext[k].min())
                extent.append(ext[k].max())
            return his, extent

def _check_data_structure(data):
    if isinstance(data, list):
        return data
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            return [data.reshape(-1, 1)]
        elif len(data.shape) == 2:
            return [data]
        elif len(data) == 3:
            return data
        else:
            raise AttributeError('''
            Data must be of shape (n_samples, n_features).''')
