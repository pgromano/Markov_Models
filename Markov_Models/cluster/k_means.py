import numpy as np
from sklearn.cluster import KMeans, k_means_
from sklearn.neighbors import DistanceMetric
from sklearn.utils import check_array, check_random_state, extmath
from copy import deepcopy

class _KMeans(KMeans):
    __doc__ = KMeans.__doc__

class _FuzzyKMeans(KMeans):
    __doc__ = KMeans.__doc__

    def __init__(self, n_clusters=8, fuzziness=2, metric='euclidean',
                 precompute_distances='auto', init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4,
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto', store_labels=False, disp=False):

        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.metric = metric
        self._distance = DistanceMetric.get_metric(metric).pairwise
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self._store_labels = store_labels
        self._disp = disp

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        # Initialize cluster centers
        x_squared_norms = extmath.row_norms(X, squared=True)
        self.cluster_centers_ = k_means_._init_centroids(
                X, self.n_clusters, self.init,
                random_state=self.random_state,
                x_squared_norms=x_squared_norms,
                init_size=None)
        centers_old = np.zeros(self.cluster_centers_.shape)

        # Expectation-Maximization
        iteration = 0
        while np.sum((centers_old - self.cluster_centers_)**2) >= self.tol:
            iteration += 1
            if iteration >= self.max_iter:
                print("""Method failed to converge after {:d} iterations. Try
                either increasing tolerance or increasing maximum number of
                iterations.""".format(self.max_iter))
                break

            # Save copy of previous iteration and update centers
            centers_old = deepcopy(self.cluster_centers_)
            self._e_step(X)
            self._m_step(X)

        if self._store_labels:
            self.labels_ = self._e_step(X)

        if self._disp:
            print("Method terminated after {:d} iterations".format(iteration))
        return self

    def _e_step(self, X):
        X = check_array(X)
        m = self.fuzziness

        # Compute distance to centroids
        D = self._distance(X, self.cluster_centers_)

        # Check for null-distances and add dummy value
        row, col = np.where(D == 0.0)
        D[row] = 1.0

        # Calculate weights
        w = (1.0 / D)**(2 / (m - 1))
        w[row] = 0.0
        w[row, col] = 1.0
        w = w / np.sum(w, axis=1)[:, None]
        return w

    def _m_step(self, X):
        D = self._e_step(X)
        w = D**self.fuzziness
        self.cluster_centers_ = np.dot(X.T, w).T
        self.cluster_centers_ /= w.sum(axis=0)[:, None]
