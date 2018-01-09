from .utils import check_sequence, check_transition_matrix
from .estimation import eigen, equilibrium, transition_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy

__all__ = ['ContinuousSequence',
           'DiscreteSequence',
           'BaseMarkovChain',
           'BaseMarkovStateModel']

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
            self.__dict__ = X.__dict__
        else:
            self.values = check_sequence(X, rank=2)
            self.n_sets = len(self.values)
            self.n_samples = [self.values[i].shape[0] for i in range(self.n_sets)]
            assert all([self.values[0].shape[1] == self.values[i].shape[1]
                        for i in range(self.n_sets)]), 'Number of features inconsistent'
            self.n_features = self.values[0].shape[1]

    def concatenate(self, features=None):
        """ Concatenates sequences

        Parameters
        ----------
        features : int, optional
            Feature to concatenate and return. None concatenates all featuress

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.concatenate
        """

        if (not hasattr(self, '_seqcat')) and (features is None):
            self._seqcat = np.concatenate([self.values[i] for i in range(self.n_sets)])
        if features is None:
            return self._seqcat
        return np.concatenate([self.values[i][:, features] for i in range(self.n_sets)])

    def histogram(self, features=None, bins=10, return_extent=False):
        """ Create histogram of sequences

        Parameters
        ----------
        features : int, optional
            Feature to build histogram. None build histogram along all
            features.
        bins : int or iterable of ints
            Number of bins to generate histogram. If bins is an iterable, it
            must be of length equal to the number of features in sequence.

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.histogramdd
        """

        his, ext = np.histogramdd(self.concatenate(features), bins=bins)
        if return_extent is True:
            extent = []
            for k in range(len(ext)):
                extent.append(ext[k].min())
                extent.append(ext[k].max())
            return his, extent
        return his

    def sample(self, size=None, features=None, replace=True):
        """ Uniformly sample from sequence data

        Parameters
        ----------
        size : int or list of ints, optional
            Size to sample from sequence
        features : int
            Features to sample along
        replace : bool
            Whether to sample with replacement

        Returns
        -------
        samples : numpy.ndarray
            Sampled values from sequences

        See Also
        --------
            numpy.random.choice
        """

        index = np.random.choice(np.arange(np.sum(self.n_samples)), size, replace)
        if len(size) > 1:
            return self.concatenate(features)[index.ravel()].reshape(size)
        return self.concatenate(features)[index]

    def split(self, train_size=0.75, val_size=None, random_state=None):
        """ Split sequence into cross-validation sets

        Parameters
        ----------
        train_size : float, optional
            Ratio of dataset size to split as training set.
        val_size : float, optional
            Ratio of dataset size to split as validation set. If None then only
            train and test sets are returned.
        random_state : float, optional
            Set seed for random splitting.

        Returns
        -------
        Xtr, Xte, Xva : float, numpy.ndarray
            The training (Xtr) and test (Xte) sets are given in the specified
            proportions. If val_size is not None, then the validation (Xva) set
            is also returned.

        See Also
        --------
            sklearn.model_selection.ShuffleSplit
        """
        assert train_size < 1.0, "Training size must be < 1.0"
        assert train_size > 0.0, "Training size must be > 0.0"
        n_obs = range(np.sum(self.n_samples))
        if val_size is None:
            test_size = 1 - train_size
            assert np.allclose(train_size + test_size, 1), "Total size must equal 1"
            for train_index, test_index in ShuffleSplit(1, test_size, random_state=random_state).split(n_obs):
                pass
            Xcat = self.concatenate()
            return Xcat[train_index], Xcat[test_index]

        test_size = 1 - (train_size + val_size)
        assert np.allclose(train_size + test_size + val_size, 1), "Total size must equal 1"

        for train_index, test_val_index in ShuffleSplit(1, val_size + test_size).split(n_obs):
            n = range(len(test_val_index))
            for i, j in ShuffleSplit(1, val_size / (val_size + test_size)).split(n):
                test_index, val_index = test_val_index[i], test_val_index[j]
        Xcat = self.concatenate()
        return Xcat[train_index], Xcat[test_index], Xcat[val_index]


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

    def __init__(self, X, n_states=None):
        if isinstance(X, DiscreteSequence):
            self.__dict__ = X.__dict__
        else:
            self.values = check_sequence(X, rank=1)
            self.n_sets = len(self.values)
            self.n_samples = [val.shape[0] for val in self.values]

            # Set label encoder
            encoder = LabelEncoder().fit(np.concatenate(self.values))
            self._values = [encoder.transform(val) for val in self.values]
            self.labels_ = encoder.classes_
            self.transform = encoder.transform
            self.inverse_transform = encoder.inverse_transform

            # Evaluate number of states
            if n_states is None:
                self.n_states = np.amax([np.amax(val) for val in self._values]) + 1
            else:
                self.n_states = n_states

    def concatenate(self):
        """ Concatenates sequences

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.concatenate
        """

        if not hasattr(self, '_seqcat'):
            self._seqcat = np.concatenate([self.values[i] for i in range(self.n_sets)], 0)
        return self._seqcat

    def counts(self, return_labels=True):
        """ Count the number of unique elements

        Parameters
        ----------
        return_labels : bool
            Whether or not to return labels of unique elements.

        Returns
        -------
        counts : int, numpy.ndarray
            The number of occurances for each unique value within sequences.
        labels : int, numpy.ndarray
            If return_labels is True, then the values of the unique states are
            returned.

        See Also
        --------
            numpy.unique
        """

        return np.unique(self.values, return_counts=return_labels)

    def one_hot(self):
        y_hot = []
        for i in range(self.n_sets):
            y_hot.append(np.zeros((self.n_samples[i], self.n_states)))
            y_hot[i][range(self.n_samples[i]), self._values[i]] = 1
        return y_hot

    def sample(self, size=None, replace=True):
        """ Uniformly sample from sequence data

        Parameters
        ----------
        size : int or list of ints, optional
            Size to sample from sequence
        replace : bool
            Whether to sample with replacement

        Returns
        -------
        samples : numpy.ndarray
            Sampled values from sequences

        See Also
        --------
            numpy.random.choice
        """

        return np.random.choice(self.concatenate(), size, replace)


class BaseMarkovChain(object):
    """ Base Model for discrete Markov chain

    Provides basic functionality for generating chains and analysis.
    """

    def populate(self, n_iter, p=None, random_state=None):
        """ Evolve population vector

        Parameters
        ----------
        n_iter : int
            The number of iterations to evolve a starting population vector.
        p : array-like iterable, shape=(n_states,), default=None
            A population vector that is evolved by the transition matrix.
            Vector must sum to 1 and have the same number of states as the
            transtition matrix.
        random_state : int, default=None
            Sets the seed for the random generation of the starting population
            vector.

        Returns
        -------
        p_n : numpy.ndarray, shape=(n_states, )
            The final population after propagating over n_iter.
        """

        n_states = self._T.shape[1]
        if p is None:
            np.random.seed(random_state)
            p = np.eye(n_states)[np.random.randint(0, n_states)]
        else:
            assert np.shape(p)[0] == n_states, "Number of states do not match"
            assert np.allclose(np.sum(p), 1), "Not a valid population vector"

        if self.n_order == 1:
            return np.matmul(p, np.linalg.matrix_power(self._T, n_iter))
        else:
            return [np.matmul(p, np.linalg.matrix_power(tmat, n_iter)) for tmat in np.split(self._T, self.n_states)]

    def simulate(self, n_samples=None, n0=None, random_state=None):
        """ Generate Markov chain from transition matrix

        Parameters
        ----------
        n_samples : int or iterable of int
            The number of samples to sample from equilibrium distribution. If
            provided list of integers, list of samples is returned with the
            same shape provided
        n0 : int
            The initial state for **all** simulations to start. If None, then
            initial states are generated at random.
        random_state : int, optional, default: None
            Set seed for random number generator.

        Returns
        -------
        simulations : list of ints
            Markov chain generated from transition matrix.
        """

        if self.n_order == 1:
            obs = equilibrium.simulate(self._T, n_samples, n0, random_state)
            return self._to_labels(obs)
        raise ValueError('Nth order Markov chains not currently supported')

    @property
    def transition_matrix(self):
        """ Transition matrix of the Markov chain """

        if hasattr(self, '_T'):
            return self._T
        else:
            self._update_transition_matrix

    @property
    def _update_transition_matrix(self):
        if hasattr(self, '_C'):
            self._T = transition_matrix(self._C, self._method,
                                        tol=self.tol, max_iter=self.max_iter)
            check_transition_matrix(self._T)
        else:
            raise ValueError("""No count matrix found""")

class BaseMarkovStateModel(object):
    """ Base Model for reversible Markov State Models

    Provides basic functionality for generating chains, kinetics, and spectral
    analysis.
    """

    def populate(self, n_iter, p=None, random_state=None):
        """ Evolve population vector

        Parameters
        ----------
        n_iter : int
            The number of iterations to evolve a starting population vector.
        p : array-like iterable, shape=(n_states,), default=None
            A population vector that is evolved by the transition matrix.
            Vector must sum to 1 and have the same number of states as the
            transtition matrix.
        random_state : int, default=None
            Sets the seed for the random generation of the starting population
            vector.

        Returns
        -------
        p_n : numpy.ndarray, shape=(n_states, )
            The final population after propagating over n_iter.
        """

        n_states = self._T.shape[0]
        if p is None:
            np.random.seed(random_state)
            p = np.eye(n_states)[np.random.randint(0, n_states)]
        else:
            assert np.shape(p)[0] == n_states, "Number of states do not match"
            assert np.allclose(np.sum(p), 1), "Not a valid population vector"

        return np.matmul(p, np.linalg.matrix_power(self._T, n_iter))

    def sample(self, n_samples=None, random_state=None):
        """ Sample from equilibrium distribution

        Parameters
        ----------
        n_samples : int or iterable of int
            The number of samples to sample from equilibrium distribution. If
            provided list of integers, list of samples is returned with the
            same shape provided
        random_state : int, optional, default: None
            Set seed for random number generator.

        Returns
        -------
        samples : list of ints
            List of states sampled from equilibrium distribution.
        """

        obs = equilibrium.sample(self.equilibrium, n_samples, random_state)
        return self._to_labels(obs)

    def simulate(self, n_samples=None, n0=None, random_state=None):
        """ Generate Markov chain from transition matrix

        Parameters
        ----------
        n_samples : int or iterable of int
            The number of samples to sample from equilibrium distribution. If
            provided list of integers, list of samples is returned with the
            same shape provided
        n0 : int
            The initial state for **all** simulations to start. If None, then
            initial states are generated at random.
        random_state : int, optional, default: None
            Set seed for random number generator.

        Returns
        -------
        simulations : list of ints
            Markov chain generated from transition matrix.
        """

        obs = equilibrium.simulate(self._T, n_samples, n0, random_state)
        return self._to_labels(obs)


    @property
    def transition_matrix(self):
        """ Transition matrix of the Markov chain """

        if hasattr(self, '_T'):
            return self._T
        else:
            self._update_transition_matrix

    @property
    def _update_transition_matrix(self):
        if hasattr(self, '_C'):
            self._T = transition_matrix(self._C, self._method,
                                        tol=self.tol, max_iter=self.max_iter)
            check_transition_matrix(self._T)
        else:
            raise ValueError("""No count matrix found""")

    @property
    def metastability(self):
        """ Crisp metastability estimated by the trace of the T matrix """

        return np.trace(self._T)

    @property
    def equilibrium(self):
        """ Returns the equilibrium (stationary) distribution """

        if not hasattr(self , '_pi'):
            self._pi = equilibrium.distribution(self._T, sparse=self._is_sparse)
        if not len(self._pi) == self.n_states:
            self._pi = equilibrium.distribution(self._T, sparse=self._is_sparse)
        return self._pi

    def eigenvalues(self, **kwargs):
        """Compute eigenvalues of transition matrix.

        Parameters
        ---------_
        k : int, optional
            The number of eigenvalues and eigenvectors desired. k must be smaller
            than N. It is not possible to compute all eigenvectors of a matrix.
        ncv : int, optional (default: min(n, max(2*k + 1, 20)))
            If the MarkovChain is initialized as a sparse, then the number of
            Lanczos vectors generated (ncv) must be greater than k; it is
            recommended that ncv > 2*k.

        Returns
        -------
        w : float, numpy.ndarray, shape=(k,)
            Array of k eigenvalues.
        """

        return eigen.values(self._T, self._is_sparse, **kwargs)

    def eigenvectors(self, method='both', **kwargs):
        """ Compute eigenvectors of transition matrix.

        Parameters
        ---------
        k : int, optional
            The number of eigenvalues and eigenvectors desired. k must be smaller
            than N. It is not possible to compute all eigenvectors of a matrix.
        ncv : int, optional (default: min(n, max(2*k + 1, 20)))
            If the MarkovChain is initialized as a sparse, then the number of
            Lanczos vectors generated (ncv) must be greater than k; it is
            recommended that ncv > 2*k.
        method : str, {'left', 'right', 'both'}, default: 'both'
            Indicates if returns left, right, or both eigenvectors.

        Returns
        -------
        L : (k, k) numpy.ndarray
            Matrix of left eigenvectors.
        R : (k, k) numpy.ndarray
            Matrix of right eigenvectors.
        """

        return eigen.vectors(self._T, method, self._is_sparse, **kwargs)

    def left_eigenvector(self, **kwargs):
        """ Compute left eigenvectors of transition matrix.

        Parameters
        ---------
        k : int, optional
            The number of eigenvalues and eigenvectors desired. k must be smaller
            than N. It is not possible to compute all eigenvectors of a matrix.
        ncv : int, optional (default: min(n, max(2*k + 1, 20)))
            If the MarkovChain is initialized as a sparse, then the number of
            Lanczos vectors generated (ncv) must be greater than k; it is
            recommended that ncv > 2*k.

        Returns
        -------
        L : (k, k) numpy.ndarray
            Matrix of left eigenvectors.
        """

        return self.eigenvectors('left', **kwargs)

    def right_eigenvector(self, **kwargs):
        """ Compute right eigenvectors of transition matrix.

        Parameters
        ---------
        k : int, optional
            The number of eigenvalues and eigenvectors desired. k must be smaller
            than N. It is not possible to compute all eigenvectors of a matrix.
        ncv : int, optional (default: min(n, max(2*k + 1, 20)))
            If the MarkovChain is initialized as a sparse, then the number of
            Lanczos vectors generated (ncv) must be greater than k; it is
            recommended that ncv > 2*k.

        Returns
        -------
        R : (k, k) numpy.ndarray
            Matrix of right eigenvectors.
        """

        return self.eigenvectors('right', **kwargs)

    def mfpt(self, origin, target=None):
        """ Mean first passage time

        Parameters
        ----------
        origin : int or iterable of ints
            Set of starting states.
        target : int or iterable of ints
            Set of target states.

        Returns
        -------
        mfpt : float, numpy.ndarray
            Mean first passage time or vector of mean first passage times.
        """

        return equilibrium.mfpt(self._T, origin, target, self._is_sparse)

    def timescales(self, k=None, **kwargs):
        """ Implied Timescales

        Parameters
        ----------
        k : int, optional
            Number of implied timescales to calculate. Must be less than the
            number of states within Markov chain. If None provided, then all
            implied timescales are calculated.
        ncv : int, optional (default: min(n, max(2*k + 1, 20)))
            If the MarkovChain is initialized as a sparse, then the number of
            Lanczos vectors generated (ncv) must be greater than k; it is
            recommended that ncv > 2*k.

        Returns
        -------
        its : float, numpy.ndarray
            Implied timescales of the transition matrix given by
            :math:`t_i = \frac{-\tau}{\ln |\lambda_i|}`

        See Also
        --------
            Markov_Models.estimation.eigen.values, scipy.linalg.eig,
            scipy.sparse.linalg.eigs
        """

        return equilibrium.timescales(self, k=k, **kwargs)
