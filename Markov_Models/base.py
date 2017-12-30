from .estimation import count_matrix, transition_matrix, eigen, equilibrium
from .utils.validation import check_sequence, check_transition_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
            self.__dict__ = X.__dict__
        else:
            self.values = check_sequence(X, rank=2)
            self.n_sets = len(self.values)
            self.n_samples = [self.values[i].shape[0] for i in range(self.n_sets)]
            assert all([self.values[0].shape[1] == self.values[i].shape[1]
                        for i in range(self.n_sets)]), 'Number of features inconsistent'
            self.n_features = self.values[0].shape[1]

    def concatenate(self, feature=None):
        """ Concatenates sequences

        Parameters
        ----------
        feature : int, optional
            Feature to concatenate and return. None concatenates all features

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.concatenate
        """

        if (not hasattr(self, '_seqcat')) and (feature is None):
            self._seqcat = np.concatenate([self.values[i] for i in range(self.n_sets)])
        if feature is None:
            return self._seqcat
        return np.concatenate([self.values[i][:, feature] for i in range(self.n_sets)])

    def histogram(self, feature=None, bins=10, return_extent=False):
        """ Create histogram of sequences

        Parameters
        ----------
        feature : int, optional
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

        his, ext = np.histogramdd(self.concatenate(feature), bins=bins)
        if return_extent is True:
            extent = []
            for k in range(len(ext)):
                extent.append(ext[k].min())
                extent.append(ext[k].max())
            return his, extent
        return his

    def sample(self, size=None, feature=None, replace=True):
        """ Uniformly sample from sequence data

        Parameters
        ----------
        size : int or list of ints, optional
            Size to sample from sequence
        feature : int
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
            return self.concatenate(feature)[index.ravel()].reshape(size)
        return self.concatenate(feature)[index]


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
            self.__dict__ = X.__dict__
        else:
            self.values = check_sequence(X, rank=1)
            encoder = LabelEncoder().fit(np.concatenate(self.values))
            self._values = [encoder.transform(val) for val in self.values]
            self.labels = encoder.classes_

            self.n_sets = len(self._values)
            self.n_samples = [val.shape[0] for val in self._values]
            self.n_states = np.amax([np.amax(val) for val in self._values]) + 1
            self.transform = encoder.transform
            self.inverse_transform = encoder.inverse_transform

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


class BaseDiscreteModel(object):
    """ Base Model for discrete Markov chains

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

    @property
    def metastability(self):
        """ Crisp metastability estimated by the trace of the T matrix """

        return np.trace(self._T)

    @property
    def equilibrium(self):
        """ Returns the equilibrium (stationary) distribution """

        if not hasattr(self , '_pi'):
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

    def score(self, objective=None):
        """ Score the Markov model

        Parameters
        ----------
        objective : {'CM', 'PR', 'GMRQ'}
            Method by which to calculates a score for the Markov chain.

        Returns
        -------
        score : float
            Score calculated by chosen objective.

        Notes
        -----
        The crisp metastability (CM) metric, given by the sum along the
        transition matrix, :math:`tr(T)`, scores the model by how metastable
        the sum of all state are.

        The persistence ratio (PR), given by the :math:`tr(T)/n`, is the ratio
        of the crisp metastability over the number of states in the chain. It
        gives the likelihood that a Markov process remain in a state, as
        opposed to transitioning.

        The generalized matrix Rayleigh quotient (GMRQ) is approximated as the
        sum of the eigenvalues of the transition matrix. [1]

        References
        ----------
        [1] McGibbon, R. T. and V. S. Pande, JCP 142, 124105 (2015),
            “Variational cross-validation of slow dynamical modes in molecular
            kinetics”

        See Also
        --------
            Markov_Models.base.DiscreteEstimator
        """

        if objective is None:
            return np.trace(self._T)
        elif objective.lower() == 'cm':
            return np.trace(self._T)
        elif objective.lower() == 'pr':
            return np.trace(self._T) / self._T.shape[0]
        elif objective.lower() == 'gmrq':
            return np.sum(self.eigenvalues())
        else:
            raise ValueError('Objective {:s} not implemented'.format(objective))


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
    observed counts from all states i to j over lag time :math:`\tau` according
    to :math:`\frac{C_{ij}(\tau)}{\sum_{j=1}^{N} C_{ij}}`. This method does
    **not** necessarily enforce detailed-balance, a requirement to Markov
    statistics.

    Symmetric estimation enforces detailed-balance by averaging the forward
    and backward transitions such that
    :math:`\bar{C}_{ij} = \bar{C}_{ji} = \frac{C_{ij} + C_{ji}}{2}`. It is not
    guaranteed that simulations whose underlying distribution obeys Markov
    statistics will exhibit a symmetric count transitions under the limit of
    ergodic sampling. The symmetrized count matrix (:math:`\bar{C}`) is row
    normalized identically to the Naive estimator. [1]

    The Prinz method employs a maximum likelihood estimation scheme detailed in
    their JCP [2] paper, which gives an excellent review of standard methods to
    estimate transition matrices from noisey time-series data.

    References
    ----------
    [1] Bowman G.R. (2014) "An Overview and Practical Guide to Building Markov
        State Models."
    [2] Prinz et al, JCP 134.17 (2011) "Markov models of molecular kinetics:
        Generation and validation."
    """

    def __init__(self, **kwargs):
        self._method = kwargs.get('method', 'prinz')
        self._is_sparse = kwargs.get('sparse', False)
        self.lag = kwargs.get('lag', 1)
        self.tol = kwargs.get('tol', 1e-4)
        self.max_iter = kwargs.get('max_iter', 1000)

    def fit(self, X):
        """ Fit Markov chain from sequence

        Parameters
        ----------
        X : DiscreteSequence or array-like, shape=(n_sets, n_samples,)
            A sequence of discrete sequences.

        Returns
        -------
        self
        """

        # Convert to DiscreteSequence class
        X = DiscreteSequence(X)

        # Set label encoder
        self._from_labels = X.transform
        self._to_labels = X.inverse_transform

        # Calculate Count and Transition Matrix
        self._C = count_matrix(X, self.lag, self._is_sparse)
        self._T = transition_matrix(self._C, self._method,
                                    tol=self.tol, max_iter=self.max_iter)

        # Validate Transition Matrix
        check_transition_matrix(self._T)
        self.n_states = self._T.shape[0]
        return self

    def score(self, X=None, objective=None, **kwargs):
        """ Score the Markov model

        Parameters
        ----------
        X : DiscreteSequence or array-like, shape=(n_sets, n_samples,)
            A sequence of discrete sequences. If given, the objective estimates a
            discrete Markov chain and scores the model accordingly. Otherwise,
            the score is based on the given model.
        objective : {'CM', 'PR', 'GMRQ'}
            Method by which to calculates a score for the Markov chain.
        **kwargs
            Key-word arguments to select conditions for estimating Markov
            chain.

        Returns
        -------
        score : float
            Score calculated by chosen objective.

        Notes
        -----
        The crisp metastability (CM) metric, given by the sum along the
        transition matrix, :math:`tr(T)`, scores the model by how metastable
        the sum of all state are.

        The persistence ratio (PR), given by the :math:`tr(T)/n`, is the ratio
        of the crisp metastability over the number of states in the chain. It
        gives the likelihood that a Markov process remain in a state, as
        opposed to transitioning.

        The generalized matrix Rayleigh quotient (GMRQ) is approximated as the
        sum of the eigenvalues of the transition matrix. [1]

        References
        ----------
        [1] McGibbon, R. T. and V. S. Pande, JCP 142, 124105 (2015),
            “Variational cross-validation of slow dynamical modes in molecular
            kinetics”

        See Also
        --------
            Markov_Models.base.DiscreteEstimator
        """

        if X is None:
            new_model = deepcopy(self)
        else:
            new_model = deepcopy(self.__class__(**kwargs).fit(X))

        if objective is None:
            return np.trace(self._T)
        elif objective.lower() == 'cm':
            return np.trace(self._T)
        elif objective.lower() == 'pr':
            return np.trace(self._T) / self._T.shape[0]
        elif objective.lower() == 'gmrq':
            return np.sum(self.eigenvalues())
        else:
            raise ValueError('Objective {:s} not implemented'.format(objective))
