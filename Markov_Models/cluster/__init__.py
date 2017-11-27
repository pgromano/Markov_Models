from .base import ContinuousClusterMixin
from .kmedoids import _KMedoids
from sklearn import cluster


__all__ = ['Affinity', 'Birch', 'DBSCAN', 'Hierarchical', 'KMeans',
           'MeanShift', 'MiniBatchKMeans', 'Spectral']


class Affinity(ContinuousClusterMixin, cluster.AffinityPropagation):
    __doc__ = cluster.AffinityPropagation.__doc__


class Birch(ContinuousClusterMixin, cluster.Birch):
    __doc__ = cluster.Birch.__doc__


class DBSCAN(ContinuousClusterMixin, cluster.DBSCAN):
    __doc__ = cluster.DBSCAN.__doc__


class Hierarchical(ContinuousClusterMixin, cluster.AgglomerativeClustering):
    __doc__ = cluster.AgglomerativeClustering.__doc__


class KMeans(ContinuousClusterMixin, cluster.KMeans):
    __doc__ = cluster.KMeans.__doc__


class KMedoids(ContinuousClusterMixin, _KMedoids):
    __doc__ = _KMedoids.__doc__


class MeanShift(ContinuousClusterMixin, cluster.MeanShift):
    __doc__ = cluster.MeanShift.__doc__


class MiniBatchKMeans(ContinuousClusterMixin, cluster.MiniBatchKMeans):
    __doc__ = cluster.KMeans.__doc__


class Spectral(ContinuousClusterMixin, cluster.SpectralClustering):
    __doc__ = cluster.SpectralClustering.__doc__
