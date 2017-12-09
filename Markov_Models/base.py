from .estimation import count_matrix, transition_matrix, eigen, equilibrium
from .utils.validation import check_array, check_transition_matrix
import numpy as np
from copy import deepcopy


class ContinuousSequence(object):
    """ Continuous sequence class to establish standard data format
    Parameters
    ----------
    X : iterable of array-like, shape=(n_sets, n_samples, n_features)
        Continuous time-series data with integer (n_sets) number of trajectory
        datasets of shape (n_samples, n_features). If ContinuousSequence is
        provided, the X attributes are inherited.

    Attributes
    ----------
    values : float, list of numpy.ndarray
        Values of all datasets.
    n_sets : int
        Number of sets of data.
    n_samples : list
        List of number of samples/observations in each set.
    n_features : int
        Number of features/dimensions in dataset. The values of n_features must
        be the same for all sets.
    """
    def __init__(self, X):
        if isinstance(X, ContinuousSequence):
            for key,item in X.__dict__.items():
                setattr(self, key, item)
        else:
            if isinstance(X, list):
                self.values = [check_array(xi, dtype=float, rank=2) for xi in X]
            elif isinstance(X, np.ndarray):
                if len(X.shape) == 2:
                    self.values = [check_array(X, dtype=float, rank=2)]
                elif len(X.shape) == 3:
                    self.values = [xi for xi in X]
                else:
                    raise ValueError('Continuous sequence must be rank 2 or 3.')
            else:
                raise ValueError('Input data unrecognized.')

            self.n_sets = len(self.values)
            self.n_samples = [self.values[i].shape[0] for i in range(self.n_sets)]
            assert all([self.values[0].shape[1] == self.values[i].shape[1]
                        for i in range(self.n_sets)]), 'Number of features inconsistent'
            self.n_features = self.values[0].shape[1]

    def concatenate(self, axis=None):
        if not hasattr(self, '_seqcat'):
            self._seqcat = np.concatenate([self.values[i] for i in range(self.n_sets)])
        if axis is None:
            return self._seqcat
        return np.concatenate([self.values[i][:, axis] for i in range(self.n_sets)])

    def histogram(self, axis=None, bins=100, return_extent=False):
        his, ext = np.histogramdd(self.concatenate(axis), bins=bins)
        if return_extent is True:
            extent = []
            for k in range(len(ext)):
                extent.append(ext[k].min())
                extent.append(ext[k].max())
            return his, extent
        return his

    def sample(self, size=None, axis=None, replace=True):
        index = np.random.choice(np.arange(np.sum(self.n_samples)), size, replace)
        if len(size) > 1:
            return self.concatenate(axis)[index.ravel()].reshape(size)
        return self.concatenate(axis)[index]


class DiscreteSequence(object):
    """ Discrete sequence class to establish standard data format
    Parameters
    ----------
    X : iterable of array-like, shape=(n_sets, n_samples)
        Discrete time-series data with integer (n_sets) number of trajectory
        datasets of shape n_samples. If DiscreteSequence is
        provided, the X attributes are inherited.

    Attributes
    ----------
    values : int, list of numpy.ndarray
        Values of all datasets.
    n_sets : int
        Number of sets of data.
    n_samples : list
        List of number of samples/observations in each set.
    """
    def __init__(self, X):
        if isinstance(X, DiscreteSequence):
            for key,item in X.__dict__.items():
                setattr(self, key, item)
        else:
            if isinstance(X, list):
                self.values = [check_array(xi, dtype=int, rank=1) for xi in X]
            elif isinstance(X, np.ndarray):
                if len(X.shape) == 1:
                    self.values = [check_array(X, dtype=int, rank=1)]
                elif len(X.shape) == 2:
                    self.values = [xi for xi in X]
                elif len(X.shape) > 2:
                    raise ValueError('Discrete sequence data must be 1D.')
            else:
                raise ValueError('Input data unrecognized.')

            self.n_sets = len(self.values)
            self.n_samples = [self.values[i].shape[0] for i in range(self.n_sets)]
            self.n_states = np.amax(self.values) + 1

    def counts(self, return_labels=True):
        states, counts = np.unique(self.values, return_counts=True)
        if return_labels:
            return states, counts
        return counts

    def sample(self, size=None, replace=True):
        return np.random.choice(self._seqcat(), size, replace)

    def _seqcat(self):
        return np.concatenate([self.values[i] for i in range(self.n_sets)], 0)


class BaseDiscreteModel(object):
    """ Base Model for discrete Markov chains

    Provides basic functionality for generating chains, kinetics, and spectral
    analysis.
    """
    def sample(self, n_samples=None):
        return equilibrium.sample(self.equilibrium, n_samples)

    def simulate(self, n_samples=None, n0=None):
        if n_samples is None:
            n_samples = 1
        return equilibrium.simulate(self._T, n_samples, n0)

    @property
    def transition_matrix(self):
        if hasattr(self, '_T'):
            return self._T

    @property
    def metastability(self):
        return np.trace(self._T)

    @property
    def equilibrium(self):
        if not hasattr(self , '_pi'):
            self._pi = equilibrium.distribution(self._T, sparse=self._is_sparse)
        return self._pi

    def eigenvalues(self, **kwargs):
        return eigen.values(self._T, self._is_sparse, **kwargs)

    def eigenvectors(self, method='both', **kwargs):
        return eigen.vectors(self._T, method, self._is_sparse, **kwargs)

    def left_eigenvector(self, method='left', **kwargs):
        return self.eigenvectors(method, **kwargs)

    def right_eigenvector(self, method='right', **kwargs):
        return self.eigenvectors(method, **kwargs)

    def mfpt(self, origin, target=None):
        return equilibrium.mfpt(self._T, origin, target, self._is_sparse)

    def timescales(self, k=None, **kwargs):
        return equilibrium.timescales(self, k=k, **kwargs)


class DiscreteModel(BaseDiscreteModel):
    """ Discrete model for Markov chains

    Parameters
    ----------
    T : float, array-like, shape=(N, N)
        Row stochastic transition matrix
    lag : int
        Number of timesteps which define length of lag-time between
        transitions.
    sparse : bool
        Whether or not sparse linear algebra methods should be used to
        compute spectral properties.

    Attributes
    ----------
    n_states : int
        Number of discrete states (or nodes) within the chain.
    """
    def __init__(self, T, **kwargs):
        self._is_sparse = kwargs.get('sparse', False)
        self.lag = kwargs.get('lag', 1)
        self._T = T
        self.n_states = self._T.shape[0]


class DiscreteEstimator(BaseDiscreteModel):
    """ Estimator for discrete Markov chains

    Parameters
    ----------
    T : float, array-like, shape=(N, N)
        Row stochastic transition matrix
    lag : int
        Number of timesteps which define length of lag-time between
        transitions.
    method : {'Prinz', 'Symmetric', 'Naive'}
        Method for estimating the transition matrix from sampled datasets.
    sparse : bool
        Whether or not sparse linear algebra methods should be used to
        compute spectral properties.

    Attributes
    ----------
    n_states : int
        Number of discrete states (or nodes) within the chain.

    Notes
    -----
    Naive estimation of the transition matrix, simply row normalizes the
    observed counts from all states i to j over lag time `\tau` according
    to `\frac{C_{ij}(\tau)}{\sum_{j=1}^{N} C_{ij}}`. This method does **not**
    necessarily enforce detailed-balance, a requirement to Markov statistics.

    Symmetric estimation enforces detailed-balance by averaging the forward
    and backward transitions such that `\bar{C}_{ij} = \bar{C}_{ji} =
    \frac{C_{ij} + C_{ji}}{2}`. It is not guaranteed that simulations whose
    underlying distribution obeys Markov statistics will exhibit a symmetric
    count transitions under the limit of ergodic sampling. The symmetrized
    count matrix (`\bar{C}`) is row normalized identically to the Naive
    estimator. [1]

    The Prinz method employs a maximum likelihood estimation scheme detailed in
    their JCP [2] paper, which gives an excellent review of standard methods to
    estimate transition matrices from noisey time-series data.

    [1] Bowman G.R. (2014) "An Overview and Practical Guide to Building Markov State Models."
    [2] Prinz et al, JCP 134.17 (2011) "Markov models of molecular kinetics: Generation and validation."
    """
    def __init__(self, **kwargs):
        self._method = kwargs.get('method', 'prinz')
        self._is_sparse = kwargs.get('sparse', False)
        self.lag = kwargs.get('lag', 1)
        self.tol = kwargs.get('tol', 1e-4)
        self.max_iter = kwargs.get('max_iter', 1000)

    def fit(self, X):
        if not isinstance(X, DiscreteSequence):
            X = DiscreteSequence(X)

        # Calculate Count and Transition Matrix
        self._C = count_matrix(X, self.lag, self._is_sparse)
        self._T = transition_matrix(self._C, self._method,
                                    tol=self.tol, max_iter=self.max_iter)

        # Validate Transition Matrix
        check_transition_matrix(self._T)
        self.n_states = self._T.shape[0]
        return self
