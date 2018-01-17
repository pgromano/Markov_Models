from .utils import check_transition_matrix
from .estimation import eigen, equilibrium, simulate, transition_matrix
import numpy as np


__all__ = ['MarkovChainMixin',
           'MarkovStateModelMixin']


class MarkovChainMixin(object):
    """ Base Model for discrete Markov chain

    Provides basic functionality for generating chains and analysis.
    """

    # TODO: Solve equilibrium distribution from dictionary transition matrix
    #def sample(self, n_samples=None, random_state=None):
    #    """ Sample from equilibrium distribution

    #    Parameters
    #    ----------
    #    n_samples : int or iterable of int
    #        The number of samples to sample from equilibrium distribution. If
    #        provided list of integers, list of samples is returned with the
    #        same shape provided
    #    random_state : int, optional, default: None
    #        Set seed for random number generator.

    #    Returns
    #    -------
    #    samples : list of ints
    #        List of states sampled from equilibrium distribution.
    #    """

    #    obs = equilibrium.sample(self.equilibrium, n_samples, random_state)
    #    return self._to_labels(obs)

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

        return simulate.from_dict(self._T, n_samples, self.n_order, n0, random_state)


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
            self._T = transition_matrix(self._C, self._method)
        else:
            raise ValueError("""No count matrix found""")


class MarkovStateModelMixin(object):
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
        if self._encode:
            return self._to_labels(obs)
        return obs

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

        obs = simulate.from_matrix(self._T, n_samples, n0, random_state)
        if self._encode:
            return self._to_labels(obs)
        return obs


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
