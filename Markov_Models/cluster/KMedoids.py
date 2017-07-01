import numpy as np
from ..analysis import count_matrix as Cmat
from ..analysis import transition_matrix as Tmat
from ..base import BaseModel
from sklearn.base import ClusterMixin, TransformerMixin
from .src import kmedoids_

class _KMedoids(BaseModel, ClusterMixin, TransformerMixin):
    '''
    k-medoids class.
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        How many medoids. Must be positive.
    distance_metric : string, optional, default: 'euclidean'
        What distance metric to use.
    clustering : {'pam'}, optional, default: 'pam'
        What clustering mode to use.
    init : {'random', 'heuristic'}, optional, default: 'heuristic'
        Specify medoid initialization.
    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting.
    random_state : int, optional, default: None
        Specify random state for the random number generator.
    '''
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.lag = kwargs.get('lag', 1)

    def fit(self, N, fraction=0.5, shuffle=True, **kwargs):
        train = self._training_set(fraction=fraction, shuffle=shuffle)
        km = kmedoids_.KMedoids(n_clusters=N, **kwargs).fit(train)
        self.centroids = km.cluster_centers_
        self.labels = [km.predict(self.data[i]) for i in range(self.n_sets)]
        self._C = self._count_matrix(lag=self.lag)
        self._T = self._transition_matrix()

        # Inherit Methods
        self.predict = km.predict

    def _count_matrix(self, lag=1):
        return Cmat(self.labels, lag=lag, sparse=self._is_sparse)

    def _transition_matrix(self, lag=None):
        if lag is not None:
            C = self._count_matrix(lag=lag)
        else:
            C = self._C
        if self._is_reversible is True:
            if self._is_force_db is True:
                return Tmat.sym_T_estimator(C)
            else:
                return Tmat.rev_T_estimator(C)
        else:
            return Tmat.nonrev_T_matrix(C)

    @property
    def count_matrix(self):
        try:
            return self._C
        except:
            raise AttributeError('''
            No instance found. Model must be fit first.''')

    @property
    def transition_matrix(self):
        try:
            return self._T
        except:
            raise AttributeError('''
            No instance found. Model must be fit first.''')

    def _training_set(self, fraction=0.1, shuffle=True):
        if fraction == 0 or fraction > 1:
            raise AttributeError('''
            Fraction must be value 0 < f <= 1.''')
        stride = int(1/fraction)
        if shuffle is True:
            idx = [np.random.permutation(np.arange(self.n_samples[i]))[::stride] for i in range(self.n_sets)]
        else:
            idx = [np.arange(self.n_samples[i])[::stride] for i in range(self.n_sets)]
        return np.concatenate([self.data[i][idx[i],:] for i in range(self.n_sets)])
