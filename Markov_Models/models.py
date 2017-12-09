from . import base as _base


def MarkovChain(T=None, **kwargs):
    if T is None:
        __doc__ = _base.DiscreteEstimator.__doc__
        return _base.DiscreteEstimator(**kwargs)
    else:
        __doc__ = _base.DiscreteModel.__doc__
        return _base.DiscreteModel(T, **kwargs)
