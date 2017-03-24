import Markov_Models as mm
import warnings
import numpy as np
import multiprocessing as mp

# TODO: Add multiprocessing to parallelize bootstrapping
class BaseMicroMSM(object):

    def __init__(self, BaseModel):
        self._base = BaseModel
        self._is_force_db = BaseModel._is_force_db
        self._is_reversible = BaseModel._is_reversible
        self._is_sparse = BaseModel._is_sparse

    def fit(self, n_states, lag=100, stride=1, method='KMeans', tol=1e-5, max_iter=500, **kwargs):
        self._base.n_microstates = n_states
        self._N = n_states
        self.lag = lag

        if method.lower() == 'kmeans':
            mm.models.cluster._KMeans(self, stride=stride, tol=tol, max_iter=max_iter, **kwargs)
        elif method.lower() == 'minibatchkmeans':
            mm.models.cluster._MiniBatchKMeans(self, stride=stride, tol=tol, max_iter=max_iter, **kwargs)

        self._C = mm.analysis.count_matrix(self.dtraj, lag=lag, sparse=self._is_sparse)
        if self._is_reversible is True:
            if self._is_force_db is True:
                self._T = mm.analysis.transition_matrix.sym_T_estimator(self._C)
            else:
                self._T = mm.analysis.transition_matrix.rev_T_estimator(self._C)
        else:
            self._T = mm.analysis.transition_matrix.nonrev_T_matrix(self._C)


    def predict(self, centroids, lag=100, method='KNeighborsClassifier', **kwargs):
        self._base.n_microstates = max(centroids.shape)
        self._N = max(centroids.shape)
        self.lag = lag
        self.centroids = centroids

        if method.lower() == 'kneighborsclassifier':
            mm.models.classifier._KNeighborsClassifier(self, centroids=centroids, **kwargs)
        if method.lower() == 'gaussiannb':
            mm.models.classifier._GaussianNB(self, centroids=centroids, **kwargs)

        self._C = mm.analysis.count_matrix(self.dtraj, lag=lag, sparse=self._is_sparse)
        if self._is_reversible is True:
            if self._is_force_db is True:
                self._T = mm.analysis.transition_matrix.sym_T_estimator(self._C)
            else:
                self._T = mm.analysis.transition_matrix.rev_T_estimator(self._C)
        else:
            self._T = mm.analysis.transition_matrix.nonrev_T_matrix(self._C)

    def _count_matrix(self, lag=1):
        return mm.analysis.count_matrix(self.dtraj, lag=lag, sparse=self._is_sparse)

    def _transition_matrix(self, lag=None):
        if lag is not None:
            C = self._count_matrix(lag=lag)
        else:
            C = self._C
        if self._is_reversible is True:
            if self._is_force_db is True:
                return mm.analysis.transition_matrix.sym_T_estimator(C)
            else:
                return mm.analysis.transition_matrix.rev_T_estimator(C)
        else:
            return mm.analysis.transition_matrix.nonrev_T_matrix(C)

    @property
    def count_matrix(self):
        return self._C

    @property
    def transition_matrix(self):
        return self._T

    @property
    def metastability(self):
        return np.diagonal(self._T).sum()

    @property
    def stationary_distribution(self):
        return mm.analysis.spectral.stationary_distribution(self._T, sparse=self._is_sparse)

    def eigenvalues(self, k=None, ncv=None):
        return mm.analysis.spectral.eigen_values(self._T, k=k, ncv=ncv, sparse=self._is_sparse, rev=self._is_reversible)

    def _eigenvectors(self, k=None, ncv=None, left=True, right=True):
        return mm.analysis.spectral.eigen_vectors(self._T, k=k, ncv=ncv, left=left, right=right, sparse=self._is_sparse, rev=self._is_reversible)

    def left_eigenvector(self, k=None, ncv=None):
        return self._eigenvectors(k=k, ncv=ncv, left=True, right=False)

    def right_eigenvector(self, k=None, ncv=None):
        return self._eigenvectors(k=k, ncv=ncv, left=False, right=True)

    def mfpt(self, origin, target):
        return mm.analysis.timescales.mfpt(self._T, origin, target)

    def timescales(self, lags=None, **kwargs):
        its = mm.analysis.timescales.ImpliedTimescaleClass(self)
        return its.implied_timescales(lags=lags, **kwargs)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key == 'lag':
                self.lag = val
                self._C = self._count_matrix(lag=val)
                self._T = self._transition_matrix(lag=val)
            elif key == 'rev':
                self._base._is_reversible = self._is_reversible = val
            elif key == 'sparse':
                self._base._is_sparse = self._is_sparse = val


class BaseMacroMSM(object):
    '''Macro level description of trajectory data, coarse grained via PCCA+
    Parameters
    ----------
    memberships :
    metastable_sets :
    metastable_clusters :
    Notes
    -----
    '''

    def __init__(self, BaseModel):
        self._base = BaseModel
        self._is_force_db = BaseModel._is_force_db
        self._is_reversible = BaseModel._is_reversible
        self._is_sparse = False
        self._micro = self._base.microstates

    def fit(self, n_macrostates, lag=None, method='PCCA'):
        self._N = n_macrostates
        self._base.n_macrostates = n_macrostates
        self.lag = self._micro.lag

        if lag is None:
            lag = self.lag

        if self._N >= self._micro._N-1:
            raise AttributeError(
                "Number of macrostates cannot be greater than N-1 of number of microstates.")
        if self._N >= 4000:
            self._is_sparse = self._base._is_sparse
            if not self._is_sparse:
                warnings.warn('''
                    Sparse methods are highly recommended for
                    microstates >= 4000! self.update(sparse=True)
                ''')

        elif self._N < 4:
            if self._is_sparse:
                raise AttributeError('''
                Too few macrostates to use the sparse method! Update to sparse=False''')

        if method.lower() == 'pcca':
            mm.analysis.coarse_grain.PCCA(self, n_macrostates, lag=lag)
        elif method.lower() == 'hmm':
            mm.analysis.coarse_grain.HMM(self, n_macrostates)
        elif method.lower() =='gmm':
            mm.analysis.coarse_grain.GMM(self, n_macrostates)
        else:
            raise AttributeError('Method '+str(method)+' is not implemented!')

        self._C = mm.analysis.count_matrix(self.dtraj, lag=lag, sparse=self._is_sparse)
        if self._is_reversible is True:
            if self._is_force_db is True:
                self._T = mm.analysis.transition_matrix.sym_T_estimator(self._C)
            else:
                self._T = mm.analysis.transition_matrix.rev_T_estimator(self._C)
        else:
            self._T = mm.analysis.transition_matrix.nonrev_T_matrix(self._C)

    # TODO: Rewrite all of this nonsense. This should predict the macrostates from a given subset of data
    #def predict(self, centroids, lag=None, method='KNeighborsClassifier', **kwargs):
    #    self._base.n_microstates = max(centroids.shape)
    #    self._N = max(centroids.shape)
    #    self.lag = self._micro.lag

    #    if lag is None:
    #        lag = self.lag

    #    if method == 'KNeighborsClassifier':
    #        mm.models.classifier._KNeighborsClassifier(self, centroids=centroids, **kwargs)
    #    if method == 'GaussianNB':
    #        mm.models.classifier._GaussianNB(self, centroids=centroids, **kwargs)

    #    self._C = mm.analysis.count_matrix(self.dtraj, lag=lag, sparse=self._is_sparse)
    #    if self._is_reversible is True:
    #        if self._is_force_db is True:
    #            self._T = mm.analysis.transition_matrix.sym_T_estimator(self._C)
    #        else:
    #            self._T = mm.analysis.transition_matrix.rev_T_estimator(self._C)
    #    else:
    #        self._T = mm.analysis.transition_matrix.nonrev_T_matrix(self._C)

    def _count_matrix(self, lag=None):
        if lag is None:
            lag = self.lag
        return mm.analysis.count_matrix(self.dtraj, lag=lag, sparse=self._is_sparse)

    def _transition_matrix(self, lag=None):
        if lag is not None:
            C = self._count_matrix(lag=lag)
        else:
            C = self._C
        if self._is_reversible is True:
            if self._is_force_db is True:
                return mm.analysis.transition_matrix.sym_T_estimator(C)
            else:
                return mm.analysis.transition_matrix.rev_T_estimator(C)
        else:
            return mm.analysis.transition_matrix.nonrev_T_matrix(C)

    @property
    def count_matrix(self):
        return self._C

    @property
    def transition_matrix(self):
        return self._T

    @property
    def metastability(self):
        return np.diagonal(self._T).sum()

    @property
    def stationary_distribution(self):
        return mm.analysis.spectral.stationary_distribution(self._T, sparse=self._is_sparse)

    def eigenvalues(self, k=None, ncv=None):
        return mm.analysis.spectral.eigen_values(self._T, k=k, ncv=ncv, sparse=self._is_sparse, rev=self._is_reversible)

    def _eigenvectors(self, k=None, ncv=None, left=True, right=True):
        return mm.analysis.spectral.eigen_vectors(self._T, k=k, ncv=ncv, left=left, right=right, sparse=self._is_sparse, rev=self._is_reversible)

    def left_eigenvector(self, k=None, ncv=None):
        return self._eigenvectors(k=k, ncv=ncv, left=True, right=False)

    def right_eigenvector(self, k=None, ncv=None, lag=None, precomputed=False):
        return self._eigenvectors(k=k, ncv=ncv, left=False, right=True)

    def mfpt(self, origin, target):
        return mm.analysis.timescales.mfpt(self._T, origin, target)

    def timescales(self, lags=None, estimate_error=False, **kwargs):
        def timescales(self, lags=None, **kwargs):
            its = mm.analysis.timescales.ImpliedTimescaleClass(self)
            return its.implied_timescales(lags=lags, **kwargs)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key == 'lag':
                self.lag = val
                self._C = self._count_matrix(lag=val)
                self._T = self._transition_matrix(lag=val)

                self._micro.lag = val
                self._micro._C = self._micro._count_matrix(lag=val)
                self._micro._T = self._micro._transition_matrix(lag=val)
            elif key == 'rev':
                self._base._is_reversible = self._is_reversible = val
            elif key == 'sparse':
                self._base._is_sparse = self._is_sparse = val
