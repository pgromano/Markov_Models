from .base import ContinuousClusterMixin as _ContinuousClusterMixin
from .kmedoids import _KMedoids
from .fuzzykmeans import _FuzzyKMeans
import numpy as _np
from sklearn import cluster as _cluster
from sklearn.utils import check_array as _check_array
from scipy.spatial.distance import cdist as _cdist


__all__ = ['Affinity', 'Birch', 'DBSCAN', 'Hierarchical', 'KMeans',
           'MeanShift', 'MiniBatchKMeans', 'Spectral']


class Affinity(_ContinuousClusterMixin, _cluster.AffinityPropagation):
    __doc__ = _cluster.AffinityPropagation.__doc__


class Birch(_ContinuousClusterMixin, _cluster.Birch):
    __doc__ = _cluster.Birch.__doc__


class DBSCAN(_ContinuousClusterMixin, _cluster.DBSCAN):
    __doc__ = _cluster.DBSCAN.__doc__


class FuzzyKMeans(_ContinuousClusterMixin, _FuzzyKMeans):
    __doc__ = _FuzzyKMeans.__doc__


class Hierarchical(_ContinuousClusterMixin, _cluster.AgglomerativeClustering):
    __doc__ = _cluster.AgglomerativeClustering.__doc__


class KMeans(_ContinuousClusterMixin, _cluster.KMeans):
    __doc__ = _cluster.KMeans.__doc__

    def membership(self, X, m=2):
        X = _check_array(X)

        # Compute distance to centroids
        D = _cdist(X, self.cluster_centers_)

        # Check for null-distances and add dummy value
        row, col = _np.where(D == 0.0)
        D[row] = 1.0

        # Calculate weights
        w = (1.0 / D)**(2 / (m - 1))
        w[row] = 0.0
        w[row, col] = 1.0
        w = w / _np.sum(w, axis=1)[:, None]
        return w


class KMedoids(_ContinuousClusterMixin, _KMedoids):
    __doc__ = _KMedoids.__doc__


class MeanShift(_ContinuousClusterMixin, _cluster.MeanShift):
    __doc__ = _cluster.MeanShift.__doc__


class MiniBatchKMeans(_ContinuousClusterMixin, _cluster.MiniBatchKMeans):
    __doc__ = _cluster.KMeans.__doc__


class Spectral(_ContinuousClusterMixin, _cluster.SpectralClustering):
    __doc__ = _cluster.SpectralClustering.__doc__
