from ..base import ContinuousSequence, DiscreteSequence
import numpy as np
from sklearn.utils import check_array
from scipy.spatial.distance import cdist


class ContinuousClusterMixin(object):
    ''' Simple tool to assist passing multiple sequences of continuous data to
    the Scikit-Learn cluster API.'''

    def fit(self, X, y=None):
        '''Fit clustering model on the data
        Parameters
        ----------
        X : array-like shape=(n_sets, n_samples, n_features)
            A sequence of continuous trajectory data.
        Returns
        -------
        self
        '''
        X = ContinuousSequence(X)
        super(ContinuousClusterMixin, self).fit(X._seqcat())

        if hasattr(self, 'labels_'):
            labels_ = []
            labels_.append(self.labels_[:X.n_samples[0]])
            if X.n_sets > 1:
                for n in range(1, X.n_sets):
                    i = X.n_samples[n - 1]
                    j = X.n_samples[n]
                    labels_.append(self.labels_[i:j])
            self.labels_ = DiscreteSequence(labels_)

        if hasattr(self, 'cluster_centers_'):
            setattr(self, 'membership', self._membership)
        return self

    def predict(self, X, y=None):
        '''Predict the closest cluster to each sample in X.
        Parameters
        ----------
        X : array-like shape=(n_sets, n_samples, n_features)
            A sequence of continuous trajectory data.
        Returns
        -------
        labels : array, shape=(n_samples,)
            Index of the cluster that each sample belongs to
        '''
        X = ContinuousSequence(X)
        labels = []
        for xi in X.values:
            labels.append(super(ContinuousClusterMixin, self).predict(xi))
        return DiscreteSequence(labels)

    def transform(self, X, y=None):
        X = ContinuousSequence(X)
        Xtr = []
        for xi in X.values:
            Xtr.append(super(ContinuousClusterMixin, self).transform(xi))
        return ContinuousSequence(Xtr)

    def fit_predict(self, X, y=None):
        self.fit(X, y).predict(X, y)

    def fit_transform(self, X, y=None):
        self.fit(X, y).transform(X, y)

    def _membership(self, X, m=2):
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError('This attribute requires self.cluster_centers_')
        X = check_array(X)

        # Compute distance to centroids
        D = cdist(X, self.cluster_centers_)

        # Check for null-distances and add dummy value
        row, col = np.where(D == 0.0)
        D[row] = 1.0

        # Calculate weights
        w = (1.0 / D)**(2 / (m - 1))
        w[row] = 0.0
        w[row, col] = 1.0
        w = w / np.sum(w, axis=1)[:, None]
        return w
