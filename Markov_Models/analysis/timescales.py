from .spectral import eigen_values, stationary_distribution
import numpy as np
from scipy.linalg import solve
from msmtools.estimation.dense import tmatrix_sampler

class ImpliedTimescaleClass(object):
    ''' Implied Timescales
    lags: int, list
    Lag time represented as number of time steps to be skipped during
    calculation of transition matrix.

    k : int, optional
    The number of eigenvalues and eigenvectors desired. k must be smaller
    than N. It is not possible to compute all eigenvectors of a matrix.

    ncv : int, option
    The number of Lanczos vectors generated ncv must be greater than k; it
    is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))

    estimate_error: bool (default: False)
    If True, the error in timescales is approximated by bootstrapping.

    n_splits: int
    Number of splits to generate from trajectory data. The data is split
    by `sklearn.model_selection.TimeSeriesSplit`, and then training sets
    are used to evaluate the standard deviation per lag time.

    n_bootstraps: int
    Number of bootstraps to to estimate error. Standard deviation for
    random replacement is generated from n_splits output.
    '''
    def __init__(self, base):
        self._is_sparse = base._is_sparse
        self._is_reversible = base._is_reversible
        self._base = base

    def implied_timescales(self, lags=None, **kwargs):
        estimate_error = kwargs.get('estimate_error', False)
        ncv = kwargs.get('ncv', None)
        n_splits = kwargs.get('n_splits', 10)
        n_bootstraps = kwargs.get('n_bootstraps', 1000)
        k = kwargs.get('k', None)
        k = _get_n_timescales(self, k)

        # Implied timescales function
        def _its(T, lag, k=k, ncv=ncv, rev=self._is_reversible, sparse=self._is_sparse):
            w = eigen_values(T, k=k, ncv=ncv, rev=rev, sparse=sparse)
            return -lag / np.log(abs(w[1:]))

        # Prepare lagtimes
        lags =  _get_lagtimes(self, lags)

        # Calculate implied timescales from full trajectory data
        its = []
        for lag in lags:
            T = self._base._transition_matrix(lag=lag)
            its.append(_its(T, lag, k=k, ncv=ncv, rev=self._is_reversible, sparse=self._is_sparse))

        # Estimate error by bootstrapping
        if estimate_error is True:
            # Use PyEMMA's Sampling reversible MSM with fixed equilibrium distribution
            error = []
            for lag in lags:
                if lag == 0:
                    error.append(np.zeros(k-1))
                else:
                    C = self._base._count_matrix(lag=lag)
                    sampler = tmatrix_sampler.TransitionMatrixSampler(C, reversible=self._is_reversible)
                    samples = sampler.sample(n_bootstraps)
                    error.append(np.array([
                        _its(T, lag, k=k, ncv=ncv,
                        rev=self._is_reversible, sparse=self._is_sparse)
                        for T in samples]).std(0))
            return np.squeeze(np.array(its)), np.squeeze(np.array(error))
        else:
            return np.squeeze(np.array(its))


def mfpt(T, origin, target):
    def mfpt_solver(T, target):
        dim = T.shape[0]
        A = np.eye(dim) - T
        A[target, :] = 0.0
        A[target, target] = 1.0
        b = np.ones(dim)
        b[target] = 0.0
        return solve(A, b)
    pi = stationary_distribution(T)

    """Stationary distribution restriced on starting set X"""
    nuX = pi[origin]
    piX = nuX / np.sum(nuX)

    """Mean first-passage time to Y (for all possible starting states)"""
    tY = mfpt_solver(T, target)

    """Mean first-passage time from X to Y"""
    tXY = np.dot(piX, tY[origin])
    return tXY

def _get_lagtimes(self, lags):
    # prepare lag as list
    if lags is None:
        lags = [self._base.lag]
    if type(lags) == int:
        lags = [lags]
    return lags

def _get_n_timescales(self, k):
    # prepare number of implied timescales
    if k == None:
        # for sparse methods must be 1 less than max
        n_its = np.amax([self._base._N - 2, 2])
    else:
        # first timescale is inifinity
        n_its = k + 1
    return n_its
