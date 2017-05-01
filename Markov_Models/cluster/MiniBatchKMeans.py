import numpy as np
from ..analysis import count_matrix as Cmat
from ..analysis import transition_matrix as Tmat
from ..base import BaseModel
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
class _MiniBatchKMeans(BaseModel, ClusterMixin, TransformerMixin):
    '''Mini-Batch K-Means clustering
    Read more in the :ref:`User Guide <mini_batch_kmeans>`.
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, optional
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.
    tol : float, default: 0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.
        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).
    batch_size : int, optional, default: 100
        Size of the mini batches.
    init_size : int, optional, default: 3 * batch_size
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.
    init : {'k-means++', 'random' or an ndarray}, default: 'k-means++'
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.
    compute_labels : boolean, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    reassignment_ratio : float, default: 0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : boolean, optional
        Verbosity mode.
    '''
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.lag = kwargs.get('lag', 1)

    def fit(self, N, fraction=0.5, shuffle=True, **kwargs):
        train = self._training_set(fraction=fraction, shuffle=shuffle)
        km = MiniBatchKMeans(n_clusters=N, **kwargs).fit(train)
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

    def _training_set(self, fraction=0.5, shuffle=True):
        if fraction == 0 or fraction > 1:
            raise AttributeError('''
            Fraction must be value 0 < f <= 1.''')
        stride = int(1/fraction)
        if shuffle is True:
            idx = [np.random.permutation(np.arange(self.n_samples[i]))[::stride] for i in range(self.n_sets)]
        else:
            idx = [np.arange(self.n_samples[i])[::stride] for i in range(self.n_sets)]
        return np.concatenate([self.data[i][idx[i],:] for i in range(self.n_sets)])
