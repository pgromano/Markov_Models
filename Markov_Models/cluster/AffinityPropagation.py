import numpy as np
from Markov_Models.base import BaseModel
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import AffinityPropagation
class _AffinityPropagation(BaseModel, ClusterMixin, TransformerMixin):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)

    def fit(self, fraction=0.5, shuffle=True, **kwargs):
        train = self._training_set(fraction=fraction, shuffle=shuffle)
        af = AffinityPropagation(**kwargs).fit(train)
        self.centroids = af.cluster_centers_
        self.labels = [af.predict(self.data[i]) for i in range(self.n_sets)]
        self.predict = af.predict

    def _training_set(self, fraction=0.5, shuffle=True):
        if fraction == 0 or fraction > 1:
            raise AttributeError('''
            Fraction must be value 0 < f <= 1.''')
        stride = [int(fraction*self.n_samples[i]) for i in range(self.n_sets)]
        if shuffle is True:
            idx = [np.random.permutation(np.arange(self.n_samples[i]))[::stride[i]] for i in range(self.n_sets)]
        else:
            idx = [np.arange(self.n_samples[i])[::stride[i]] for i in range(self.n_sets)]
        return np.concatenate([self.data[i][idx[i],:] for i in range(self.n_sets)])
