from .base import ContinuousClusterMixin as _ContinuousClusterMixin
from .fuzzykmeans import _FuzzyKMeans
from .kmedoids import _KMedoids
from sklearn import cluster as _cluster


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


class KMedoids(_ContinuousClusterMixin, _KMedoids):
    __doc__ = _KMedoids.__doc__


class KMeans(_ContinuousClusterMixin, _cluster.KMeans):
    __doc__ = _cluster.KMeans.__doc__


class MeanShift(_ContinuousClusterMixin, _cluster.MeanShift):
    __doc__ = _cluster.MeanShift.__doc__


class MiniBatchKMeans(_ContinuousClusterMixin, _cluster.MiniBatchKMeans):
    __doc__ = _cluster.KMeans.__doc__


class Spectral(_ContinuousClusterMixin, _cluster.SpectralClustering):
    __doc__ = _cluster.SpectralClustering.__doc__
