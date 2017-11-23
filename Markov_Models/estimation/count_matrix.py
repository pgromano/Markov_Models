import numpy as np
from scipy.sparse import coo_matrix


def _count(X, lag, n_states, sparse):
    i = X[:-lag]
    j = X[lag:]

    C = coo_matrix((np.ones(i.size), (i, j)), shape=(n_states, n_states))
    if sparse is True:
        return C.tocsr()
    return C.toarray()


def count_matrix(X, lag=1, sparse=False):
    C = [_count(X.values[i], lag, X.n_states, sparse) for i in range(X.n_sets)]
    return np.sum(C, axis=0)