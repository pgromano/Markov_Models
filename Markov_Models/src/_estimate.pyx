import numpy as np

def count_matrix(dtraj, nsamples, lag, nstates):
    cdef int i, j, n, t

    maxsamples = dtraj.shape[0]
    nsets = dtraj.shape[1]
    C = np.zeros((nstates, nstates), dtype=int)

    for n from 0 <= n < nsets:
        for t from 0 <= t < nsamples[n]:
            if t+lag <= nsamples[n]-1:
                i = dtraj[t,n]
                j = dtraj[t+lag, n]
                C[i,j] += 1
    return C

def crisp_assignment(dtraj, PCCA_sets):
    cdef int i
    nsamples = dtraj.shape[0]
    PCCA_dtraj = np.zeros(nsamples, dtype=int)

    for i from 0 <= i < nsamples:
        PCCA_dtraj[i] = PCCA_sets[dtraj[i]] - 1
    return PCCA_dtraj
