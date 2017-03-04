r"""This is a modification of MSMtools module which implements the countmatrix estimation functionality
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
"""

import numpy as np
from scipy.sparse import coo_matrix

def _count_matrix(dtraj, lag, nstates, sparse):
    row = dtraj[:-lag]
    col = dtraj[lag:]
    data = np.ones(row.size)

    C = coo_matrix((data, (row, col)), shape=(nstates, nstates))
    if sparse is True:
        return C.tocsr()
    else:
        return C.toarray()

def count_matrix(dtraj, lag=1, sparse=False):
    nstates = np.amax(dtraj)+1
    nsets = len(dtraj)

    C = [_count_matrix(dtraj[i], lag, nstates, sparse) for i in range(nsets)]
    return np.sum(C, axis=0)

# TODO :: Incorporate multiprocessing
