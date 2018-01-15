import numpy as np
from scipy.sparse import coo_matrix

__all__ = ['count_matrix']

def _count(X, lag, n_states, sparse):
    i = X[:-lag]
    j = X[lag:]

    C = coo_matrix((np.ones(i.size), (i, j)), shape=(n_states, n_states))
    if sparse is True:
        return C.tocsr()
    return C.toarray()


def count_matrix(X, lag=1, sparse=False):
    X.n_states = np.amax([np.amax(val) for val in X.values]) + 1
    C = [_count(val, lag, X.n_states, sparse) for val in X.values]
    return np.sum(C, axis=0)


def count_vectorizer(X, n_order=1, lag=1):
    C = {}
    for x in X.values:
        for k in range(len(x) - n_order * lag):
            # Define initial and final states
            i = tuple(x[[k + n * lag for n in range(n_order)]])
            j = x[k + n_order * lag]

            # Check if initial state has been sampled
            if i not in C:
                C[i] = {}

            # Check if final state has been sampled from initial
            if j not in C[i]:
                C[i][j] = 0

            # Add 1 to number of state visits
            C[i][j] += 1
    return C
