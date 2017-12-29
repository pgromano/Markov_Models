from . import base as _base
from .k_means import _KMeans, _FuzzyKMeans
from .kmedoids import _KMedoids
from sklearn import cluster as _cluster


__all__ = ['Affinity', 'Birch', 'DBSCAN', 'Hierarchical', 'KMeans',
           'MeanShift', 'MiniBatchKMeans', 'Spectral']


class Affinity(_base.ContinuousClusterMixin, _cluster.AffinityPropagation):
    __doc__ = _cluster.AffinityPropagation.__doc__


class Birch(_base.ContinuousClusterMixin, _cluster.Birch):
    __doc__ = _cluster.Birch.__doc__


class DBSCAN(_base.ContinuousClusterMixin, _cluster.DBSCAN):
    __doc__ = _cluster.DBSCAN.__doc__


class FuzzyKMeans(_base.ContinuousClusterMixin, _FuzzyKMeans):
    __doc__ = _FuzzyKMeans.__doc__


class Hierarchical(_base.ContinuousClusterMixin, _cluster.AgglomerativeClustering):
    __doc__ = _cluster.AgglomerativeClustering.__doc__


class KMedoids(_base.ContinuousClusterMixin, _KMedoids):
    __doc__ = _KMedoids.__doc__


class KMeans(_base.ContinuousClusterMixin, _KMeans):
    __doc__ = _KMeans.__doc__


class MeanShift(_base.ContinuousClusterMixin, _cluster.MeanShift):
    __doc__ = _cluster.MeanShift.__doc__


class MiniBatchKMeans(_base.ContinuousClusterMixin, _cluster.MiniBatchKMeans):
    __doc__ = _cluster.MiniBatchKMeans.__doc__


class Spectral(_base.ContinuousClusterMixin, _cluster.SpectralClustering):
    __doc__ = _cluster.SpectralClustering.__doc__
