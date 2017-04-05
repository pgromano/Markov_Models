import numpy as np
from ..analysis import count_matrix as Cmat
from ..analysis import transition_matrix as Tmat
from ..base import BaseModel
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.mixture import GaussianMixture
class _GaussianMixture(BaseModel, ClusterMixin, TransformerMixin):
    '''Gaussian Mixture.
    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.
    .. versionadded:: 0.18
    *GaussianMixture*.
    Read more in the :ref:`User Guide <gmm>`.
    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).
    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    reg_covar : float, defaults to 0.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.
    means_init: array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.
    precisions_init: array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    random_state : RandomState or an int seed, defaults to None.
        A random number generator instance.
    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.
    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.
    verbose_interval : int, default to 10.
        Number of iteration done before the next print.
    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.
    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.
    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
    lower_bound_ : float
        Log-likelihood of the best fit of EM.
    '''

    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.lag = kwargs.get('lag', 1)

    def fit(self, N, fraction=0.5, shuffle=True, **kwargs):
        train = self._training_set(fraction=fraction, shuffle=shuffle)
        gm = GaussianMixture(n_components=N, **kwargs).fit(train)
        self.centroids, self.weights, self.covariances = gm.means_, gm.weights_, gm.covariances_
        self.labels = [gm.predict(self.data[i]) for i in range(self.n_sets)]
        self._C = self._count_matrix(lag=self.lag)
        self._T = self._transition_matrix()

        # Inherit Methods
        self.predict = gm.predict
        self.predict_prob = gm.predict_proba
        self.sample = gm.sample
        self.score = gm.score
        self.score_samples = gm.score_samples

    def _count_matrix(self, lag=1):
        return Cmat(self.labels, lag=lag, sparse=self._is_sparse)

    def _transition_matrix(self, lag=None):
        if lag is not None:
            C = self._count_matrix(lag=lag)
        else:
            C = self._C
        if self._is_reversible is True:
            if self._is_force_db is True:
                return Tmat.sym_T_estimator(C)
            else:
                return Tmat.rev_T_estimator(C)
        else:
            return Tmat.nonrev_T_matrix(C)

    @property
    def count_matrix(self):
        try:
            return self._C
        except:
            raise AttributeError('''
            No instance found. Model must be fit first.''')

    @property
    def transition_matrix(self):
        try:
            return self._T
        except:
            raise AttributeError('''
            No instance found. Model must be fit first.''')

    def _training_set(self, fraction=0.5, shuffle=True):
        if fraction == 0 or fraction > 1:
            raise AttributeError('''
            Fraction must be value 0 < f <= 1.''')
        stride = [int(fraction*self.n_samples[i]) for i in range(self.n_sets)]
        if shuffle is True:
            idx = [np.random.permutation(np.arange(self.n_samples[i]))[::stride[i]] for i in range(self.n_sets)]
        else:
            idx = [np.arange(self.n_samples[i])[::stride[i]] for i in range(self.n_sets)]
        return np.concatenate([self.data[i][idx[i],:] for i in range(self.n_sets)])
