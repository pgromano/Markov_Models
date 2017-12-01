import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate(double[:, :] T, np.int n_samples, np.int n0, np.int n_sets):
    cdef long n_states = T.shape[0]
    cdef long[:] X = np.zeros(n_samples).astype(np.int)
    cdef int i, j, n

    if n0 is None:
        X[0] = np.random.randint(0, n_states, size=1).astype(np.int)
    else:
        if n0 >= n_states:
            raise ValueError('Initial state outside of network!!!')
        X[0] = n0

    for n in range(1, n_samples):
        if (n_sets != 1) and (n % np.int(n_samples / n_sets) == 0):
            X[n] = n0
        else:
            X[n] = np.random.choice(n_states, size=1, p=T[X[n - 1], :]).astype(np.int)
    return X
