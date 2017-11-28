from .estimation import count_matrix, transition_matrix, eigen, equilibrium
from .utils.validation import check_array, check_transition_matrix
import numpy as np
from copy import deepcopy


class ContinuousSequence(object):
    def __init__(self, X):
        if isinstance(X, ContinuousSequence):
            self.values = X.values
        else:
            if isinstance(X, list):
                self.values = [check_array(xi, dtype=float, rank=2) for xi in X]
            elif isinstance(X, np.ndarray):
                if len(X.shape) == 2:
                    raise ValueError('''For continuous sequences, a 2D array is
                                        vague with respect to number of
                                        datasets, samples, and features.
                                        Please, pass a 3D array.''')
                elif len(X.shape) == 3:
                    self.values = [xi for xi in X]
                elif len(X.shape) > 3:
                    raise ValueError('Continuous sequence data must be 3D.')
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
        return np.random.choice(self.concatenate(axis), size, replace)


class DiscreteSequence(object):
    def __init__(self, X):
        if isinstance(X, DiscreteSequence):
            self.values = X.values
        else:
            if isinstance(X, list):
                self.values = [check_array(xi, dtype=int, rank=1) for xi in X]
            elif isinstance(X, np.ndarray):
                if len(X.shape) == 2:
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


class DiscreteModel(object):
    def __init__(self, T, **kwargs):
        self._is_sparse = kwargs.get('sparse', False)
        self.lag = kwargs.get('lag', 1)
        self._T = T

    def sample(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        return equilibrium.sample(self._T, n_samples)

    @property
    def transition_matrix(self):
        if hasattr(self, '_T'):
            return self._T

    @property
    def metastability(self):
        return np.diagonal(self._T).sum()

    @property
    def equilibrium(self):
        return equilibrium.distribution(self._T, sparse=self._is_sparse)

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

    def timescales(self, k=None):
        return equilibrium.timescales(self, k)


class DiscreteEstimator(object):
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

    def sample(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        return equilibrium.sample(self._T, n_samples)

    @property
    def count_matrix(self):
        if hasattr(self, '_C'):
            return self._C
        raise AttributeError('''
            No instance found. Model must be fit first.''')

    @property
    def transition_matrix(self):
        if hasattr(self, '_T'):
            return self._T
        else:
            if hasattr(self, '_C'):
                self._T = transition_matrix(self._C, self._method,
                                            tol=self.tol,
                                            max_iter=self.max_iter)
                return self._T
            else:
                raise AttributeError('''
                    No instance found. Model must be fit first.''')

    @property
    def metastability(self):
        return np.diagonal(self._T).sum()

    @property
    def equilibrium(self):
        return equilibrium.distribution(self._T, sparse=self._is_sparse)

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

    def timescales(self, k=None):
        return equilibrium.timescales(self, k)
