from . import base as _base


def MarkovChain(T=None, **kwargs):
    '''
    Markov Chain estimated from discrete sequence trajectory.
    '''
    if T is None:
        return _base.DiscreteEstimator(**kwargs)
    else:
        return _base.DiscreteModel(T, **kwargs)
