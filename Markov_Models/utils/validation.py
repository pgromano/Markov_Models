import numpy as np


__all__ = ['check_irreducible',
           'check_sequence',
           'check_transition_matrix', 
           'check_equilibrium']


def check_irreducible(X):
    if not len(np.arange(X.n_states)) == len(X.counts()[0]):
        raise ValueError('Sequence does not sample all states.')


def check_sequence(X, rank=1, dtype=None):
    def iscontainer(X):
        if isinstance(X, (list, tuple)):
            if isinstance(X[0], (list, tuple, np.ndarray)):
                return True
        if isinstance(X, np.ndarray):
            if np.ndim(X) == rank + 1:
                return True
        return False

    # Check X is iterable
    if not isinstance(X, (list, tuple, np.ndarray)):
        raise ValueError("""Sequence must be iterable of type list, tuple, or numpy.ndarray.""")

    # Check whether X is a container of many sequences
    if iscontainer(X):
        assert np.ndim(X[0]) == rank, "Sequence must be rank {:d}".format(rank)
        if dtype is not None:
            return [np.array(xi, dtype=dtype) for xi in X]
        return [np.array(xi) for xi in X]
    else:
        assert np.ndim(X) == rank, "Sequence must be rank {:d}".format(rank)
        if dtype is not None:
            return [np.array(X, dtype=dtype)]
        return [np.array(X)]


def check_transition_matrix(T):
    assert np.allclose(T.sum(1), 1), 'Not valid transition matrix'


def check_equilibrium(pi, n=None):
    assert len(pi.shape) == 1, "Must be 1D vector"
    assert pi.sum() == 1, "Distribution must sum to 1"
    if n is not None:
        assert len(pi) == n, "Number of states must be {:d}".format(n)
    return pi
