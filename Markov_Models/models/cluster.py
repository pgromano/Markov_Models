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
    alg = KMeans(n_clusters=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.data[i][::stride, :]
                   for i in range(self._base.n_sets)])))
        self.dtraj = [alg.predict(csr_matrix(self._base.data[i])) for i in range(self._base.n_sets)]
    else:
        alg.fit(np.concatenate([self._base.data[i][::stride, :]
                   for i in range(self._base.n_sets)]))
        self.dtraj = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    self.centroids = alg.cluster_centers_


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

    if self._N > 300:
        batch_size = int(self._N / 2)
    else:
        batch_size = 100

    alg = MiniBatchKMeans(n_clusters=self._N, batch_size=batch_size)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.data[i][::stride, :]
                                for i in range(self._base.n_sets)])))
        self.dtraj = [alg.predict(csr_matrix(self._base.data[i]))
                                for i in range(self._base.n_sets)]
    else:
        alg.fit(np.concatenate([self._base.data[i][::stride, :]
                               for i in range(self._base.n_sets)]))
        self.dtraj = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    self.centroids = alg.cluster_centers_

from sklearn.cluster import MeanShift
def _MeanShift(self, stride=1, **kwargs):
    '''Mean shift clustering using a flat kernel.
    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.
    Seeding is performed using a binning technique for scalability.
    Read more in the :ref:`User Guide <mean_shift>`.
    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth used in the RBF kernel.
        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).
    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.
    bin_seeding : boolean, optional
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        default value: False
        Ignored if seeds argument is not None.
    min_bin_freq : int, optional
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds. If not defined, set to 1.
    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.
    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.
    Notes
    -----
    Scalability:
    Because this implementation uses a flat kernel and
    a Ball Tree to look up members of each kernel, the complexity will tend
    towards O(T*n*log(n)) in lower dimensions, with n the number of samples
    and T the number of points. In higher dimensions the complexity will
    tend towards O(T*n^2).
    Scalability can be boosted by using fewer seeds, for example by using
    a higher value of min_bin_freq in the get_bin_seeds function.
    Note that the estimate_bandwidth function is much less scalable than the
    mean shift algorithm and will be the bottleneck if it is used.
    References
    ----------
    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.
    '''
    alg = MeanShift()
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.data[i][::stride, :]
                                for i in range(self._base.n_sets)])))
        self.dtraj = [alg.predict(csr_matrix(self._base.data[i]))
                                for i in range(self._base.n_sets)]
    else:
        alg.fit(np.concatenate([self._base.data[i][::stride, :]
                               for i in range(self._base.n_sets)]))
        self.dtraj = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    self.centroids = alg.cluster_centers_

from sklearn.cluster import Birch
def _Birch(self, stride=1, **kwargs):
    '''Implements the Birch clustering algorithm.
    Every new sample is inserted into the root of the Clustering Feature
    Tree. It is then clubbed together with the subcluster that has the
    centroid closest to the new sample. This is done recursively till it
    ends up at the subcluster of the leaf of the tree has the closest centroid.
    Read more in the :ref:`User Guide <birch>`.
    Parameters
    ----------
    threshold : float, default 0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started.
    branching_factor : int, default 50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        the node has to be split. The corresponding parent also has to be
        split and if the number of subclusters in the parent is greater than
        the branching factor, then it has to be split recursively.
    n_clusters : int, instance of sklearn.cluster model, default 3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples. If None, this final
        clustering step is not performed and the subclusters are returned
        as they are. If a model is provided, the model is fit treating
        the subclusters as new samples and the initial data is mapped to the
        label of the closest subcluster. If an int is provided, the model
        fit is AgglomerativeClustering with n_clusters set to the int.
    compute_labels : bool, default True
        Whether or not to compute labels for each fit.
    copy : bool, default True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.
    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.
    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.
    subcluster_centers_ : ndarray,
        Centroids of all subclusters read directly from the leaves.
    subcluster_labels_ : ndarray,
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.
    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf
    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch
    '''
    alg = Birch(n_clusters=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.trj[i][::stride, :]
                                for i in range(self._base.nSim)])))
        self.centroids = alg.subcluster_centers_
        self._dtrj = [alg.predict(csr_matrix(self._base.trj[i]))
                                for i in range(self._base.nSim)]
    else:
        alg.fit(np.concatenate([self._base.trj[i][::stride, :]
                               for i in range(self._base.nSim)]))
        self.centroids = alg.subcluster_centers_
        self._dtrj = [alg.predict(self._base.trj[i]) for i in range(self._base.nSim)]


from sklearn.cluster import DBSCAN
def _DBSCAN(self, stride=1, **kwargs):
    '''Perform DBSCAN clustering from vector array or distance matrix.
    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.
    Read more in the :ref:`User Guide <dbscan>`.
    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.
        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.
    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
    components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.
    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    Notes
    -----
    See examples/cluster/plot_dbscan.py for an example.
    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n).
    Sparse neighborhoods can be precomputed using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>`
    with ``mode='distance'``.
    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
    '''
    alg = DBSCAN(min_samples=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.trj[i][::stride, :]
                                for i in range(self._base.nSim)])))
    else:
        alg.fit(np.concatenate([self._base.trj[i][::stride, :]
                               for i in range(self._base.nSim)]))
    self.centroids = alg.components_
    self._N = self.centroids.shape[0]
    assign(self)
