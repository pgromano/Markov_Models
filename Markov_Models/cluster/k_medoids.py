import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.neighbors import DistanceMetric
from sklearn.utils import check_array, check_random_state
from copy import deepcopy


class _KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """ K-Medoids Class

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of medoids, k, to fit the model.
    metric : string, optional, default: 'euclidean'
        Distance metric to use, see `sklearn.neighbors.DistanceMetric`.
    init : {'random', 'heuristic'}, optional, default: 'heuristic'
        Specify initialization method for medoids. The 'random' option randomly
        generates initial centroids from training data, and 'heuristic' selects
        the k first data points that minimize the distance to remaining points.
    max_iter : int, optional, default : 300
        The maximum number of iterations when fitting model.
    random_state : int, optional, default: None
        Set seed for random number generator.
    disp : bool, option, default: False
        Print output statistics on fit.
    """

    def __init__(self, n_clusters=10, metric='euclidean', init='heuristic',
                 max_iter=300, random_state=None, disp=False):

        self.n_clusters = n_clusters
        self.metric = metric
        self._distance = DistanceMetric.get_metric(metric).pairwise
        self.init = init
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self._disp = disp

    def fit(self, X, y=None):
        """ K-medoids fit method

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training data to fit model
        y : None
            Necessary for Scikit-Learn BaseEstimator
        """

        # Apply distance metric to get the distance matrix
        X = check_array(X)
        D = self._distance(X)

        # Old medoids will be stored here for reference
        medoids = self._k_init(D, self.n_clusters)
        medoids_old = np.zeros((self.n_clusters,))

        iteration = 0
        while not np.all(medoids_old == medoids):
            iteration += 1
            if iteration >= self.max_iter:
                print("""Method failed to converge after {:d} iterations. Try
                either increasing tolerance or increasing maximum number of
                iterations.""".format(self.max_iter))
                break

            # Save copy of previous iteration and update medoids
            medoids_old = deepcopy(medoids)
            labels = self._k_labels(D, medoids)
            labels, medoids = self._k_update(D, labels, medoids)

        # Set fit attributes
        self.labels_ = labels
        self.cluster_centers_ = X[medoids]

        if self._disp:
            print('Method terminated after {:d} iterations'.format(iteration))
        return self

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data to transform.
        Returns
        -------
        X_new : array, shape=(n_samples, n_clusters)
            X transformed in the new space.
        """

        assert hasattr(self, "cluster_centers_"), 'Model must be fit'
        X = check_array(X)
        return self._distance(X, Y=self.cluster_centers_)

    def predict(self, X):
        Xtr = self.transform(X)
        return np.argmin(Xtr, axis=1)

    def inertia(self, X):
        Xtr = self.transform(X)
        return np.sum(Xtr.sum(1))

    def _k_init(self, D, n_clusters):
        if self.init == 'random':
            medoids = self.random_state_.permutation(D.shape[0])[:n_clusters]
        elif self.init == 'heuristic':
            medoids = list(np.argsort(np.sum(D, axis=1))[:n_clusters])
        else:
            raise ValueError("init = {:s}, not implemented".format(self.init))
        return medoids

    def _k_labels(self, D, medoids):
        """ Labels medoids by minimized distance """
        return np.argmin(D[medoids, :], axis=0)

    def _k_update(self, D, labels, medoids):
        """ In-place update of the medoid indices """
        for k in range(self.n_clusters):
            if sum(labels == k) == 0:
                print("Cluster {:d} is empty!".format(k))

            # Current cost
            cost = np.sum(D[medoids[k], labels == k])

            # Distances between medoids
            D_in = D[labels == k, :]
            D_in = D_in[:, labels == k]

            # Cost between each point
            all_costs = np.sum(D_in, axis=1)

            # Find the smallest cost in k
            index = np.argmin(all_costs)
            min_cost = all_costs[index]

            # Update to medoids which minimize cost
            if min_cost < cost:
                medoids[k] = np.where(labels == k)[0][index]
        return labels, medoids
