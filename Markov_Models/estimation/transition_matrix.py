from . import _mle_tmat_prinz
import numpy as np
from sklearn.preprocessing import normalize


__all__ = ['transition_matrix']


def transition_matrix(C, method='Naive', **kwargs):
    """ Helper function to run transition matrix calculation by method """

    if method.lower() == 'prinz':
        return prinz(C, **kwargs)
    elif method.lower() == 'symmetric':
        return symmetric(C)
    elif method.lower() == 'naive':
        return naive(C)
    elif method.lower() == 'dictionary':
        return dictionary(C)
    else:
        raise ValueError("""Normalization method {:s} is not currently
        implemented""".format(method))


def naive(C):
    """ Naive estimator (non-reversible)

    Notes
    -----
    Naive estimation of the transition matrix, simply row normalizes the
    observed counts from all states i to j over lag time :math:`\tau` according
    to :math:`\frac{C_{ij}(\tau)}{\sum_{j=1}^{N} C_{ij}}`. This method does
    **not** necessarily enforce detailed-balance, a requirement to Markov
    statistics.
    """
    return normalize(C, norm="l1")


def symmetric(C):
    """ Symmetric estimator (reversible)

    Notes
    -----
    Symmetric estimation enforces detailed-balance by averaging the forward
    and backward transitions such that
    :math:`\bar{C}_{ij} = \bar{C}_{ji} = \frac{C_{ij} + C_{ji}}{2}`. It is not
    guaranteed that simulations whose underlying distribution obeys Markov
    statistics will exhibit a symmetric count transitions under the limit of
    ergodic sampling. The symmetrized count matrix (:math:`\bar{C}`) is row
    normalized identically to the Naive estimator. [1]

    References
    ----------
    [1] Bowman G.R. (2014) "An Overview and Practical Guide to Building Markov State Models."
    """
    Cs = 0.5 * (C + C.T)
    return normalize(Cs, norm="l1")


def prinz(C, tol=1e-4, max_iter=1000):
    """ Maximum Likelihood Estimator developed by Prinz et al

    Notes
    -----
    The Prinz method employs a maximum likelihood estimation scheme detailed in
    their JCP [2] paper, which gives an excellent review of standard methods to
    estimate transition matrices from noisey time-series data.

    References
    ----------
    [2] Prinz et al, JCP 134.17 (2011) "Markov models of molecular kinetics: Generation and validation."
    """
    return np.asarray(_mle_tmat_prinz.transition_matrix(C, tol, max_iter))

def dictionary(C):
    T = {}
    for row_key, row_val in C.items():
        T[row_key] = {}
        weight = 0
        for col_key, col_val in row_val.items():
            weight += col_val
        for col_key, col_val in row_val.items():
            T[row_key][col_key] = C[row_key][col_key] / weight
    return T
