import numpy as np
from ..analysis import count_matrix as Cmat
from ..analysis import transition_matrix as Tmat
from ..base import BaseModel
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
class _KMeans(BaseModel, ClusterMixin, TransformerMixin):
    '''K-Means clustering
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    verbose : int, default 0
        Verbosity mode.
    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    '''
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.lag = kwargs.get('lag', 1)

    def fit(self, N, fraction=0.5, shuffle=True, **kwargs):
        train = self._training_set(fraction=fraction, shuffle=shuffle)
        km = KMeans(n_clusters=N, **kwargs).fit(train)
        self.centroids, self._inertia = km.cluster_centers_, km.inertia_
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
