import numpy as np
from .src import _mle_tmat_prinz

def nonrev_T_matrix(C):
    return C/C.sum(1)[:,None]

def sym_T_estimator(C):
    Cs = 0.5*(C+C.T)
    return Cs/Cs.sum(1)[:,None]

def rev_T_estimator(C, tol=1e-4, max_iter=1000):
    return np.asarray(_mle_tmat_prinz.transition_matrix(C, tol, max_iter))

def mcmc_T_estimator(C):
    pass
