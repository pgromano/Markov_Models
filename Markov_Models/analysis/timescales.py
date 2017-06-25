from .spectral import eigen_values, stationary_distribution

import numpy as np
from scipy.linalg import solve
from pyemma.msm import its as _implied_timescales
from msmtools.analysis.dense.mean_first_passage_time import mfpt as _dense_mfpt
from msmtools.analysis.sparse.mean_first_passage_time import mfpt as _sparse_mfpt
from multiprocessing import cpu_count


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
        self.labels = base.labels
        self.lag = base.lag

    def implied_timescales(self, lags=None, **kwargs):
        k = kwargs.get('k', None)
        errors = kwargs.get('errors', None)
        n_samples = kwargs.get('n_samples', 100)
        n_jobs = kwargs.get('n_jobs', cpu_count())
        if n_jobs == -1 or n_jobs == None:
            n_jobs = cpu_count()

        # Prepare lagtimes
        lags =  _get_lagtimes(self, lags)

        # Calculate implied timescales from full trajectory data
        its = _implied_timescales(self.labels, lags=lags, nits=k,
                                  reversible=self._is_reversible,
                                  errors=errors, nsamples=n_samples,
                                  n_jobs=n_jobs)

        if errors == None:
            return its.timescales
        elif errors == 'bayes':
            return its.timescales, its.sample_std

def mfpt(T, origin, target, sparse=False):
    if sparse:
        return _sparse_mfpt(T, origin, target)
    else:
        return _dense_mfpt(T, origin, target)

def _get_lagtimes(self, lags):
    # prepare lag as list
    if lags is None:
        lags = [self.lag]
    if type(lags) == int:
        lags = [lags]
    elif type(lags) == np.ndarray:
        lags = list(lags)
    return lags
