from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def transition_matrix(double[:,:] C, np.float tol, np.int max_iter):
    cdef long n_states = C.shape[0]
    cdef double[:,:] T = np.zeros((n_states, n_states)).astype(np.float64)
    cdef double[:,:] X = np.zeros((n_states, n_states)).astype(np.float64)
    cdef double[:] xs = np.zeros(n_states).astype(np.float64)
    cdef double[:] cs = np.zeros(n_states).astype(np.float64)
    cdef double likelihood = 0
    cdef double likelihood_prev = 1e6
    cdef int i, j, k, n_iter
    cdef double val_a, val_b, val_c, tmp

    # Uncondintional transition matrix $X_{i, j} = \pi_i T_{i, j}$
    for i in range(n_states):
        for j in range(n_states):
            X[i, j] = C[i, j] + C[j, i]

    for i in range(n_states):
        xs[i] = 0
        cs[i] = 0
        for j in range(n_states):
            xs[i] = xs[i] + X[i, j]
            cs[i] = cs[i] + C[i, j]
        if xs[i] == 0 or cs[i] == 0:
            print('Null populated states!')
            break

    n_iter = 0
    while abs(likelihood - likelihood_prev) >= tol:
        likelihood_prev = likelihood
        likelihood = 0

        n_iter += 1
        if n_iter > max_iter:
            print('Method failed to converge after ' + str(max_iter) + ' n_iters. Try either increasing tolerance or increasing maximum number of n_iters.')
            break

        for i in range(n_states):
            tmp = X[i, i]
            if cs[i] - C[i, i] > 0:
                X[i, i] = C[i, i]*(xs[i] - X[i, i]) / (cs[i] - C[i, i])
            xs[i] = xs[i] + (X[i, i] - tmp)

            xs[i] = 0
            for j in range(n_states):
                xs[i] = xs[i] + X[i, j]
            if xs[i] < 0:
                print('ERROR! x.sum(1) = '+str(xs[i]))

            if X[i, i] > 0:
                likelihood = likelihood + C[i, i]*np.log(X[i, i] / xs[i])

        for i in range(n_states):
            for j in range(n_states):
                # a parameter value
                val_a = cs[i] - C[i, j] + cs[j] - C[j, i]

                # b parameter value
                val_b = cs[i] * (xs[j] - X[i, j]) \
                        + cs[j] * (xs[i] - X[i, j]) \
                        - (C[i, j] + C[j, i]) * (xs[i] + xs[j] - 2 * X[i, j])

                # c parameter value
                val_c = -(C[i, j] + C[j, i])*(xs[i] - X[i, j])*(xs[j] - X[i, j])

                if val_c > 0:
                    print(r'ERROR! Value c > 0.')

                if val_a == 0:
                    tmp = X[j, i]
                else:
                    tmp = (-val_b + np.sqrt(val_b*val_b - 4*val_a*val_c)) / (2*val_a)

                xs[i] = xs[i] + (tmp - X[i, j])
                xs[j] = xs[j] + (tmp - X[j, i])

                X[i, j] = X[j, i] = tmp

                xs[i] = 0
                for k in range(n_states):
                    xs[i] = xs[i] + X[i,k]

                xs[j] = 0
                for k in range(n_states):
                    xs[j] = xs[j] + X[j,k]

                if X[i, j] > 0:
                    likelihood = likelihood + C[i, j] * np.log(X[i, j] / xs[i]) \
                                + C[j, i] * np.log(X[j, i] / xs[j])

    for i in range(n_states):
        for j in range(n_states):
            T[i, j] = X[i, j] / xs[i]

    return T
