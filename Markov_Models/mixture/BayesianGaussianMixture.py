import numpy as np
from ..analysis import count_matrix as Cmat
from ..analysis import transition_matrix as Tmat
from ..base import BaseModel
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture
class _BayesianGaussianMixture(BaseModel, ClusterMixin, TransformerMixin):
    '''Variational Bayesian estimation of a Gaussian mixture.
    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.
    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.
    .. versionadded:: 0.18
    *BayesianGaussianMixture*.
    Read more in the :ref:`User Guide <bgmm>`.
    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
        String describing the type of covariance parameters to use.
        Must be one of::
            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).
    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.
    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
    n_init : int, defaults to 1.
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.
    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.
    weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
        String describing the type of the weight concentration prior.
        Must be one of::
            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).
    weight_concentration_prior : float | None, optional.
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.
    mean_precision_prior : float | None, optional.
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed. Smaller
        values concentrate the means of each clusters around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it's set to 1.
    mean_prior : array-like, shape (n_features,), optional
        The prior on the mean distribution (Gaussian).
        If it is None, it's set to the mean of X.
    degrees_of_freedom_prior : float | None, optional.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.
    covariance_prior : float or array-like, optional
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::
                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'
    random_state: RandomState or an int seed, defaults to None.
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
    '''

    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.lag = kwargs.get('lag', 1)

    def fit(self, N, fraction=0.5, shuffle=True, **kwargs):
        train = self._training_set(fraction=fraction, shuffle=shuffle)
        bm = BayesianGaussianMixture(n_components=N, **kwargs).fit(train)
        self.centroids, self.weights, self.covariances = bm.means_, bm.weights_, bm.covariances_
        self.labels = [bm.predict(self.data[i]) for i in range(self.n_sets)]
        self._C = self._count_matrix(lag=self.lag)
        self._T = self._transition_matrix()

        # Inherit Methods
        self.predict = bm.predict
        self.predict_prob = bm.predict_proba
        self.sample = bm.sample
        self.score = bm.score
        self.score_samples = bm.score_samples

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
