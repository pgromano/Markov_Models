from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def transition_matrix(double[:,:] C, np.float tol, np.int max_iteration):
    cdef long nstates = C.shape[0]
    cdef double[:,:] T = np.zeros((nstates, nstates)).astype(np.float64)
    cdef double[:,:] X = np.zeros((nstates, nstates)).astype(np.float64)
    cdef double[:] xs = np.zeros(nstates).astype(np.float64)
    cdef double[:] cs = np.zeros(nstates).astype(np.float64)
    cdef double liklihood = 0
    cdef double liklihood_prev = 1e6
    cdef int i, j, k, iteration
    cdef double val_a, val_b, val_c, tmp

    # Uncondintional transition matrix $X_{i,j} = \pi_i T_{i,j}$
    for i in range(nstates):
        for j in range(nstates):
            X[i,j] = C[i,j] + C[j,i]

    for i in range(nstates):
        xs[i] = 0
        cs[i] = 0
        for j in range(nstates):
            xs[i] = xs[i] + X[i,j]
            cs[i] = cs[i] + C[i,j]

    iteration = 0
    while abs(liklihood-liklihood_prev) >= tol:
        liklihood_prev = liklihood
        liklihood = 0

        iteration += 1
        if iteration > max_iteration:
            print('Method failed to converge after '+str(max_iteration)+' iterations. Try either increasing tolerance or increasing maximum number of iterations.')
            break

        for i in range(nstates):
            tmp = X[i,i]
            if cs[i] - C[i,i] > 0:
                X[i,i] = C[i,i]*(xs[i] - X[i,i]) / (cs[i] - C[i,i])
            xs[i] = xs[i] + (X[i,i] - tmp)

            xs[i] = 0
            for j in range(nstates):
                xs[i] = xs[i] + X[i,j]
            if xs[i] < 0:
                print('ERROR! x.sum(1) = '+str(xs[i]))

            if X[i,i] > 0:
                liklihood = liklihood + C[i,i]*np.log(X[i,i]/xs[i])

        for i in range(nstates):
            for j in range(nstates):
                val_a = cs[i] - C[i,j] + cs[j] - C[j,i]
                val_b = cs[i]*(xs[j] - X[i,j]) + cs[j]*(xs[i] - X[i,j]) - (C[i,j] + C[j,i])*(xs[i] + xs[j] -2*X[i,j])
                val_c = -(C[i,j] + C[j,i])*(xs[i] - X[i,j])*(xs[j] - X[i,j])

                if val_c > 0:
                    print(r'ERROR! Value c > 0.')

                if val_a == 0:
                    tmp = X[j,i]
                else:
                    tmp = (-val_b + np.sqrt(val_b*val_b - 4*val_a*val_c)) / (2*val_a)

                xs[i] = xs[i] + (tmp - X[i,j])
                xs[j] = xs[j] + (tmp - X[j,i])

                X[i,j] = X[j,i] = tmp

                xs[i] = 0
                for k in range(nstates):
                    xs[i] = xs[i] + X[i,k]

                xs[j] = 0
                for k in range(nstates):
                    xs[j] = xs[j] + X[j,k]

                if X[i,j] > 0:
                    liklihood = liklihood + C[i,j]*np.log(X[i,j]/xs[i]) + C[j,i]*np.log(X[j,i]/xs[j])

    for i in range(nstates):
        for j in range(nstates):
            T[i,j] = X[i,j]/xs[i]

    return T
