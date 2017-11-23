from ..base import DiscreteSequence
from ..estimation import count_matrix, transition_matrix, eigen
import numpy as np


def aic(T):
    return 2 * T.shape[0] - 2 * np.log(np.product(T.ravel()))


def relaxation_time(X, lags=1, n_lags=10, bootstrap=None,
               db=False, rev=True,
               sparse=False, ncv=None):

    # Check Lag Values
    if isinstance(lags, (np.ndarray, list)):
        n_lags = len(lags)
    else:
        lags = np.arange(n_lags) * lags

    # Convert Training Data to Sequence
    X = DiscreteSequence(X)

    if bootstrap is not None:
        t = _relaxation_times(X, lags, rev, db, sparse, ncv)
        straps = _bootstrap_rt(X, lags, bootstrap, rev, db, sparse, ncv)
        return t, straps
    else:
        return _relaxation_times(X, lags, rev, db, sparse, ncv)


def _relaxation_times(X, lags, rev, db, sparse, ncv):
    t = np.zeros(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag == 0:
            t[i] = 0
        else:
            # Compute Count Matrix
            C = count_matrix(X, lag, sparse)

            # Compute Transition Matrix
            if rev is True:
                if db is True:
                    T = transition_matrix.symmetric(C)
                else:
                    T = transition_matrix.prinz(C)
            else:
                T = transition_matrix.naive(C)

            # Compute Timescales
            w = eigen.values(T, sparse, k=2, ncv=ncv)[1]
            t[i] = -lag / np.log(abs(w))
    return t


def _bootstrap_rt(X, lags, n_straps, rev, db, sparse, ncv):
    n_samples = np.sum(X.n_samples)
    Xs = np.split(X.sample(size=n_samples * n_straps), n_straps, axis=0)
    straps = []
    for strap in range(n_straps):
        t = np.zeros(len(lags), dtype=np.float64)
        for i, lag in enumerate(lags):
            if lag == 0:
                t[i] = 0
            else:
                # Compute Count Matrix
                C = count_matrix(DiscreteSequence(Xs[strap]), lag, sparse)

                # Compute Transition Matrix
                if rev is True:
                    if db is True:
                        T = transition_matrix.symmetric(C)
                    else:
                        T = transition_matrix.prinz(C)
                else:
                    T = transition_matrix.naive(C)

                # Compute Timescales
                w = eigen.values(T, sparse, k=2, ncv=ncv)[1]
                t[i] = -lag / np.log(abs(w))
        straps.append(t)
    return straps
