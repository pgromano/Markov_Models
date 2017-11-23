from .base import DiscreteEstimator, DiscreteModel


def MarkovChain(T=None, **kwargs):
    '''
    Markov Chain estimated from discrete sequence trajectory.
    '''
    if T is None:
        return DiscreteEstimator(**kwargs)
    else:
        return DiscreteModel(T, **kwargs)
