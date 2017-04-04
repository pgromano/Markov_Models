import numpy as np
from scipy.sparse import csr_matrix

from sklearn.cluster import KMeans
def _KMeans(self, stride=1, tol=1e-5, max_iter=500, **kwargs):
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

    # Prepare clustering method
    alg = KMeans(n_clusters=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)

    # Shuffle data for training set
    idx = [np.random.permutation(np.arange(self._base.n_samples[i]))[::stride] for i in range(self._base.n_sets)]
    train = np.concatenate([self._base.data[i][idx[i],:] for i in range(self._base.n_sets)])

    # Fit training set Dense/Sparse and predict datasets in discrete coordinates
    if self._is_sparse:
        alg.fit(csr_matrix(train))
        labels = [alg.predict(csr_matrix(self._base.data[i])) for i in range(self._base.n_sets)]
    else:
        alg.fit(train)
        labels = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    centroids = alg.cluster_centers_
    return centroids, labels


from sklearn.cluster import MiniBatchKMeans
def _MiniBatchKMeans(self, stride=1, **kwargs):
    '''Mini-Batch K-Means clustering
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
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point (if compute_labels is set to True).
    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition (if compute_labels is set to True). The inertia is
        defined as the sum of square distances of samples to their nearest
        neighbor.
    See also
    --------
    KMeans
        The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.
    Notes
    -----
    See http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
    '''

    # Test batch size for safe input
    if self._N > 300:
        batch_size = int(self._N / 2)
    else:
        batch_size = 100

    # Prepare clustering method
    alg = MiniBatchKMeans(n_clusters=self._N, batch_size=batch_size)
    for key,val in kwargs.items():
        setattr(alg,key,val)

    # Shuffle data for training set
    idx = [np.random.permutation(np.arange(self._base.n_samples[i]))[::stride] for i in range(self._base.n_sets)]
    train = np.concatenate([self._base.data[i][idx[i],:] for i in range(self._base.n_sets)])

    # Fit training set Dense/Sparse and predict datasets in discrete coordinates
    if self._is_sparse:
        alg.fit(csr_matrix(train))
        labels = [alg.predict(csr_matrix(self._base.data[i])) for i in range(self._base.n_sets)]
    else:
        alg.fit(train)
        labels = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    centroids = alg.cluster_centers_
    return centroids, labels
