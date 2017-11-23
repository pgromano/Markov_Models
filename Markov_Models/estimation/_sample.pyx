import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def sample(double[:,:] T, np.int n_samples):
    cdef long n_states = T.shape[0]
    cdef long[:] X = np.zeros(n_samples + 1).astype(np.int)
    cdef int i, j, n

    X[0] = np.random.randint(0, n_states+1, size=1).astype(np.int)
    for n in range(1, n_samples + 1):
        X[n] = np.random.choice(n_states, size=1, p=T[X[n-1],:]).astype(np.int)
    return X[1:]
