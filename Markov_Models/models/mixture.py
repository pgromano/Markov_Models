import numpy as np
from scipy.sparse import csr_matrix

from sklearn.mixture import GaussianMixture
def _GaussianMixture(self, stride=1, **kwargs):
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
    alg = GaussianMixture(n_components=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.trj[i][::stride, :]
                                for i in range(self._base.nSim)])))
        self._dtrj = [alg.predict(csr_matrix(self._base.trj[i]))
                                for i in range(self._base.nSim)]
    else:
        alg.fit(np.concatenate([self._base.trj[i][::stride, :]
                               for i in range(self._base.nSim)]))
        self._dtrj = [alg.predict(self._base.trj[i]) for i in range(self._base.nSim)]
    self.centroids = alg.means_

from sklearn.mixture import BayesianGaussianMixture
def _BayesianGaussianMixture(self, stride=1, **kwargs):
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
        The shape depends on ``covariance_type``::
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
        time. The shape depends on ``covariance_type``::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.
    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of inference.
    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::
            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).
        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.
    weight_concentration_ : array-like, shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).
    mean_precision_prior : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed.
        Smaller values concentrate the means of each clusters around
        `mean_prior`.
    mean_precision_ : array-like, shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).
    means_prior_ : array-like, shape (n_features,)
        The prior on the mean distribution (Gaussian).
    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).
    degrees_of_freedom_ : array-like, shape (n_components,)
        The number of degrees of freedom of each components in the model.
    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::
            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'
    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.
    References
    ----------
    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <http://www.springer.com/kr/book/9780387310732>`_
    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_
    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <http://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_
    '''
    alg = BayesianGaussianMixture(n_components=self._N)
    for key,val in kwargs.items():
        setattr(alg,key,val)
    if self._is_sparse:
        alg.fit(csr_matrix(np.concatenate([self._base.trj[i][::stride, :]
                                for i in range(self._base.nSim)])))
        self._dtrj = [alg.predict(csr_matrix(self._base.trj[i]))
                                for i in range(self._base.nSim)]
    else:
        alg.fit(np.concatenate([self._base.trj[i][::stride, :]
                               for i in range(self._base.nSim)]))
        self._dtrj = [alg.predict(self._base.trj[i]) for i in range(self._base.nSim)]
    self.centroids = alg.means_
