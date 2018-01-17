from .base import MarkovChainMixin, MarkovStateModelMixin
from .estimation import count_matrix, count_vectorizer
from .utils import DiscreteSequence
import numpy as np
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy


__all__ = ['MarkovChain', 'MarkovStateModel']


class MarkovChain(MarkovChainMixin):
    """ Estimator for Discrete Markov Chains

    Parameters
    ----------
    T : float, nested dict, option
        Row stochastic transition matrix. Outer dictionary must contain tuples
        of length n_order. Notice if this is not passed, n_order is set as the
        length of the first key. Inner dictionary must define transition
        probabilities from initial sequence of length n_order. If transition
        matrix not provided, the method must be fit from sequence data.
    n_order : int, option
        The order of the Markov chain
    lag : int, optional
        Number of timesteps which define length of lag-time between
        transitions.

    See Also
    --------
        numpy.random.choice
        Markov_Models.models.MarkovStateModels
    """

    def __init__(self, T=None, n_order=1, lag=1):
        if T is not None:
            self._T = T
            self.n_order = len(list(T.keys())[0])
        else:
            self.n_order = n_order
        self.lag = lag
        self._method = "Dictionary"

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

        # Calculate Count and Transition Matrix
        self._C = count_vectorizer(X, self.n_order, self.lag)
        self._update_transition_matrix
        return self


class MarkovStateModel(MarkovStateModelMixin):
    """ Estimator for Markov State Models

    Parameters
    ----------
    T : float, array-like, shape=(N, N), optional
        Row stochastic transition matrix. If not provided, the method must be
        fit from sequence data.
    lag : int, optional
        Number of timesteps which define length of lag-time between
        transitions.
    n_order : int, optional
        Order of the Markov chain.
    labels : array-like, optional
        List of values to label the states within the Markov chain.
    method : {'Prinz', 'Symmetric', 'Naive'}, optional
        Method for estimating the transition matrix from sampled datasets.
    tol : float, optional
        The tolerance threshold for MLE fitting of transition matrix.
    max_iter : int, optional
        The maximum number of threshold for MLE fitting of transition matrix.
    sparse : bool
        Whether or not sparse linear algebra methods should be used to
        compute spectral properties.

    Attributes
    ----------
    n_states : int
        Number of discrete states (or nodes) within the chain.

    See Also
    --------
        numpy.random.choice
        sklearn.preprocessing.LabelEncoder
        Markov_Models.models.MarkovStateModels

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

    def __init__(self, T=None, lag=1, method='Prinz', tol=1e-4, max_iter=1000,
                 sparse=False, labels=None):
        self._method = method
        self._is_sparse = sparse
        self.lag = lag
        self.tol = tol
        self.max_iter = max_iter
        self.labels_ = labels
        self._encode = False

        if T is not None:
            self._T = T
            self.n_states = T.shape[1]
            assert T.shape[0] == T.shape[1], "MSM must be first order Markov chain"

        # Check Labels
        if self.labels_ is not None:
            if T is not None:
                assert len(self.labels_) == T.shape[0], "T matrix does not match labels"
            encoder = LabelEncoder().fit(self.labels_)
            self._encode = True
            self._from_labels = encoder.transform
            self._to_labels = encoder.inverse_transform

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
        if self._encode:
            X = DiscreteSequence(X, encoder=self._from_labels)
        else:
            X = DiscreteSequence(X)

        # Calculate Count and Transition Matrix
        self._C = count_matrix(X, self.lag, self._is_sparse)
        self.n_states = self._C.shape[1]
        self._update_transition_matrix
        return self

    def score(self, X=None, objective=None, **kwargs):
        """ Score the Markov model

        Parameters
        ----------
        X : DiscreteSequence or array-like, shape=(n_sets, n_samples,)
            A sequence of discrete sequences. If given, the objective estimates a
            Markov state model and scores the model accordingly. Otherwise, the
            score is based on the given model.
        objective : {'crisp', 'crisp_norm', 'GMRQ'}
            Method by which to calculates a score for the Markov state model.
        **kwargs
            Key-word arguments to select conditions for estimating Markov state
            model.

        Returns
        -------
        score : float
            Score calculated by chosen objective.

        Notes
        -----
        The crisp metastability (crisp) metric, given by the sum along the
        transition matrix, :math:`tr(T)`, scores the model by how metastable
        the sum of all state are.

        The persistence ratio (crisp_norm), given by the :math:`tr(T)/n`, is the ratio
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
            return np.trace(new_model._T)
        elif objective.lower() == 'crisp':
            return np.trace(new_model._T)
        elif objective.lower() == 'crisp_norm':
            return np.trace(new_model._T) / new_model._T.shape[1]
        elif objective.lower() == 'gmrq':
            return np.sum(new_model.eigenvalues())
        else:
            raise ValueError('Objective {:s} not implemented'.format(objective))
