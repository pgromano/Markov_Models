import numpy as np
from . import _mle_tmat_prinz


def transition_matrix(C, method, **kwargs):
    if method == 'prinz':
        return prinz(C, **kwargs)
    elif method == 'symmetric':
        return symmetric(C)
    elif method == 'naive':
        return naive(C)


def naive(C):
    return C / C.sum(1)[:, None]


def symmetric(C):
    Cs = 0.5 * (C + C.T)
    return Cs / Cs.sum(1)[:, None]


def prinz(C, tol=1e-4, max_iter=1000):
    return np.asarray(_mle_tmat_prinz.transition_matrix(C, tol, max_iter))


def mcmc(C):
    pass
