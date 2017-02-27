import numpy as np
from scipy.sparse import csr_matrix

from hmmlearn.hmm import GaussianHMM
def _GaussianHMM(self, stride=1, **kwargs):
    '''Hidden Markov Model with Gaussian emissions.
    Parameters
    ----------
    n_components : int
        Number of states.
    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of
        * "spherical" --- each state uses a single variance value that
          applies to all features;
        * "diag" --- each state uses a diagonal covariance matrix;
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix;
        * "tied" --- all states use **the same** full covariance matrix.
        Defaults to "diag".
    min_covar : float
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.
    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.
    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.
    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".
    random_state: RandomState or an int seed
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.
    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.
    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.
    covars\_ : array
        Covariance parameters for each state.
        The shape depends on ``covariance_type``::
            (n_components, )                        if 'spherical',
            (n_features, n_features)                if 'tied',
            (n_components, n_features)              if 'diag',
            (n_components, n_features, n_features)  if 'full'
    '''
    alg = GaussianHMM(n_components=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.data[i][::stride, :]
                                for i in range(self._base.n_sets)])))
        self.dtraj = [alg.predict(csr_matrix(self._base.data[i]))
                                for i in range(self._base.n_sets)]
    else:
        alg.fit(np.concatenate([self._base.data[i][::stride, :]
                               for i in range(self._base.n_sets)]))
        self.dtraj = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    self.centroids = alg.means_
