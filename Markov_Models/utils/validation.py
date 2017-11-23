import numpy as np


def check_irreducible(X):
    if not len(np.arange(X.n_states)) == len(X.counts()[0]):
        raise ValueError('Sequence does not sample all states.')


def check_array(X, dtype=None, rank=1):
    X = np.array(X)
    if not len(X.shape) == rank:
        raise ValueError('Array rank must be {:d}'.format(rank))
    if dtype is not None:
        return X.astype(dtype)
    return X


def check_transition_matrix(T):
    pass


def check_equilibrium(pi, n=None):
    assert len(pi.shape) == 1, "Must be 1D vector"
    assert pi.sum() == 1, "Distribution must sum to 1"
    if n is not None:
        assert len(pi) == n, "Number of states must be {:d}".format(n)
    return pi
