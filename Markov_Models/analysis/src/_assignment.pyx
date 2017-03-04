import numpy as np

def crisp_assignment(dtraj, PCCA_sets):
    cdef int i
    cdef long nsamples = dtraj.shape[0]
    cdef long[:] PCCA_dtraj = np.zeros(nsamples, dtype=int)

    for i in range(nsamples):
        PCCA_dtraj[i] = PCCA_sets[dtraj[i]] - 1
    return PCCA_dtraj
