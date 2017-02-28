import Markov_Models as mm
from msmtools.estimation.dense import tmatrix_sampler
import warnings
import numpy as np
import multiprocessing as mp

# TODO: Add multiprocessing to parallelize bootstrapping
class BaseMicroMSM(object):

    def __init__(self, MSM):
        self._base = MSM
        self._is_sparse = MSM._is_sparse
        self._is_reversible = MSM._is_reversible

    def fit(self, n_states, lag=100, stride=1, method='KMeans', tol=1e-5, max_iter=500, **kwargs):
        self._base.n_microstates = n_states
        self._N = n_states
        self._base.lag = self.lag = lag

        if method == 'HMM':
            mm.models.hmm._GaussianHMM(self, stride=stride, tol=tol, max_iter=max_iter, **kwargs)
        elif method == 'KMeans':
            mm.models.cluster._KMeans(self, stride=stride, tol=tol, max_iter=max_iter, **kwargs)
        elif method == 'MiniBatchKMeans':
            mm.models.cluster._MiniBatchKMeans(self, stride=stride, tol=tol, max_iter=max_iter, **kwargs)

    def predict(self, centroids, lag=100, method='KNeighborsClassifier', **kwargs):
        self._base.n_microstates = max(centroids.shape)
        self._N = max(centroids.shape)
        self._base.lag = self.lag = lag
        self.centroids = centroids

        if method == 'KNeighborsClassifier':
            mm.models.classifier._KNeighborsClassifier(self, centroids=centroids, **kwargs)
        if method == 'GaussianNB':
            mm.models.classifier._GaussianNB(self, centroids=centroids, **kwargs)

    def count_matrix(self, lag=None):
        if lag is None:
            lag = self.lag
        self._C = mm.analyze.trajectory.CMatrix(self.dtraj, lag=lag)
        return self._C

    def transition_matrix(self, lag=None):
        if lag is None:
            lag = self.lag
        self._T = mm.analyze.trajectory.TransitionMatrix(self.dtraj, lag=lag, rev=self._is_reversible)
        return self._T

    def metastability(self, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return np.diagonal(T).sum()

    def eigenvalues(self, k=None, ncv=None, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.EigenValues(T, k=k, ncv=ncv, sparse=self._is_sparse, rev=self._is_reversible)

    def _eigenvectors(self, k=None, ncv=None, lag=None, left=True, right=True, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.EigenVectors(T, k=k, ncv=ncv, left=left, right=right, sparse=self._is_sparse, rev=self._is_reversible)

    def left_eigenvector(self, k=None, ncv=None, lag=None, precomputed=False):
        return self._eigenvectors(lag=lag, k=k, ncv=ncv, left=True, right=False, precomputed=precomputed)

    def right_eigenvector(self, k=None, ncv=None, lag=None, precomputed=False):
        return self._eigenvectors(lag=lag, k=k, ncv=ncv, left=False, right=True, precomputed=precomputed)

    def stationary_distribution(self, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.StationaryDistribution(T, sparse=self._is_sparse)

    def mfpt(self, origin, target, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.mfpt(T, origin, target)

    def timescales(self, lags=None, k=1, estimate_error=False, **kwargs):
        ''' Implied Timescales
        lags: int, list
            Lag time represented as number of time steps to be skipped during
            calculation of transition matrix.

        k : int, optional
            The number of eigenvalues and eigenvectors desired. k must be smaller
            than N. It is not possible to compute all eigenvectors of a matrix.

        ncv : int, option
            The number of Lanczos vectors generated ncv must be greater than k; it
            is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))

        estimate_error: bool (default: False)
            If True, the error in timescales is approximated by bootstrapping.

        n_splits: int
            Number of splits to generate from trajectory data. The data is split
            by `sklearn.model_selection.TimeSeriesSplit`, and then training sets
            are used to evaluate the standard deviation per lag time.

        n_bootstraps: int
            Number of bootstraps to to estimate error. Standard deviation for
            random replacement is generated from n_splits output.
        '''
        for kw,arg in kwargs.items():
            if kw == 'n_bootstraps':
                n_bootstraps = arg

        #n_splits=10
        #n_bootstraps=1000
        # Prepare number of implied timescales
        k = _n_its(self, k)

        # Prepare lags array
        lags = _lags(self, lags)

        # Calculate implied timescales from full trajectory data
        its = _Timescales(self, lags, k, **kwargs)

        # Estimate error by bootstrapping
        if estimate_error is True:
            # Use PyEMMA's Sampling reversible MSM with fixed equilibrium distribution
            error = []
            for lag in lags:
                if lag == 0:
                    error.append(np.zeros(k-1))
                else:
                    sampler = tmatrix_sampler.TransitionMatrixSampler(self.count_matrix(lag=lag),
                            reversible=self._is_reversible)
                    samples = sampler.sample(n_bootstraps)
                    error.append(np.array([
                        mm.analyze.trajectory.Timescales(
                            T, lag, k=k, sparse=self._is_sparse, rev=self._is_reversible, **kwargs)
                        for T in samples]).std(0))
            return its, np.array(error)
        else:
            return its

    def voronoi(self, stride=1, clusters=None, pbc=None, bins=100, method='full'):
        idx = [np.random.permutation(np.arange(self._base.n_samples[i]))[::stride] for i in range(self._base.n_sets)]
        train = np.concatenate([self._base.data[i][idx[i],:] for i in range(self._base.n_sets)])
        labels = np.concatenate([self.dtraj[i][idx[i]] for i in range(self._base.n_sets)])
        centroids = self.centroids
        if method == 'full':
            return mm.analyze.voronoi.FullVoronoi(train, centroids, clusters=clusters, pbc=pbc, bins=bins)
        elif method == 'classify':
            raise AttributeError('Note yet implemented')
            #return mm.analyze.voronoi.ClassifyVoronoi(train, labels, bins=bins)


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

    def __init__(self, MSM):
        self._base = MSM
        self._is_sparse = False
        self._is_reversible = MSM._is_reversible

    def fit(self, n_macrostates, lag=None, method='PCCA'):
        self._N = n_macrostates
        self._base.n_macrostates = n_macrostates
        self._micro = self._base.microstates

        if lag is None:
            self.lag = self._base.lag
        else:
            self._base.lag = lag
            self._micro.lag = lag

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

        if method == 'PCCA':
            mm.analyze.coarse_grain.PCCA(self, n_macrostates, lag=lag)
            mm.analyze.coarse_grain.assign(self)
        else:
            raise AttributeError('Method '+str(method)+' is not implemented!')

    def predict(self, centroids, lag=100, method='KNeighborsClassifier', **kwargs):
        self._base.n_microstates = max(centroids.shape)
        self._N = max(centroids.shape)
        self._base.lag = self.lag = lag
        self.centroids = centroids

        if method == 'KNeighborsClassifier':
            mm.models.classifier._KNeighborsClassifier(self, centroids=centroids, **kwargs)
        if method == 'GaussianNB':
            mm.models.classifier._GaussianNB(self, centroids=centroids, **kwargs)

    def transform(self, centroids=None, clusters=None):
        # TODO: This should take given centroids and set sets for all centroids
        # transform trajectory data
        if centroids is None:
            centroids = self._micro.centroids
        if clusters is None:
            clusters = [[i] for i in range(self._micro._N)]
        # CODE THAT BUILDS DTRAJ
        pass

    def count_matrix(self, lag=None):
        if lag is None:
            lag = self.lag
        self._C = mm.analyze.trajectory.CMatrix(self.dtraj, lag=lag)
        return self._C

    def transition_matrix(self, lag=None):
        if lag is None:
            lag = self.lag
        self._T = mm.analyze.trajectory.TransitionMatrix(self.dtraj, lag=lag, rev=self._is_reversible)
        return self._T

    def metastability(self, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return np.diagonal(T).sum()

    def eigenvalues(self, k=None, ncv=None, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.EigenValues(T, k=k, ncv=ncv, sparse=self._is_sparse, rev=self._is_reversible)

    def _eigenvectors(self, k=None, ncv=None, lag=None, left=True, right=True, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.EigenVectors(T, k=k, ncv=ncv,
                                             left=left, right=right, sparse=self._is_sparse, rev=self._is_reversible)

    def left_eigenvector(self, k=None, ncv=None, lag=None, precomputed=False):
        return self._eigenvectors(lag=lag, k=k, ncv=ncv, left=True, right=False, precomputed=precomputed)

    def right_eigenvector(self, k=None, ncv=None, lag=None, precomputed=False):
        return self._eigenvectors(lag=lag, k=k, ncv=ncv, left=False, right=True, precomputed=precomputed)

    def stationary_distribution(self, lag=None, key=0, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.StationaryDistribution(T, sparse=self._is_sparse)

    def mfpt(self, origin, target, lag=None, precomputed=False):
        T = _GetTransitionMatrix(self, lag, precomputed)
        return mm.analyze.spectral.mfpt(T, origin, target)

    def timescales(self, lags=None, k=1, estimate_error=False, **kwargs):
        ''' Implied Timescales
        lags: int, list
            Lag time represented as number of time steps to be skipped during
            calculation of transition matrix.

        k : int, optional
            The number of eigenvalues and eigenvectors desired. k must be smaller
            than N. It is not possible to compute all eigenvectors of a matrix.

        ncv : int, option
            The number of Lanczos vectors generated ncv must be greater than k; it
            is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))

        estimate_error: bool (default: False)
            If True, the error in timescales is approximated by bootstrapping.

        n_splits: int
            Number of splits to generate from trajectory data. The data is split
            by `sklearn.model_selection.TimeSeriesSplit`, and then training sets
            are used to evaluate the standard deviation per lag time.

        n_bootstraps: int
            Number of bootstraps to to estimate error. Standard deviation for
            random replacement is generated from n_splits output.
        '''
        for kw,arg in kwargs.items():
            if kw == 'n_bootstraps':
                n_bootstraps = arg

        #n_splits=10
        #n_bootstraps=1000
        # Prepare number of implied timescales
        k = _n_its(self, k)

        # Prepare lags array
        lags = _lags(self, lags)

        # Calculate implied timescales from full trajectory data
        its = _Timescales(self, lags, k, **kwargs)

        # Estimate error by bootstrapping
        if estimate_error is True:
            # Use PyEMMA's Sampling reversible MSM with fixed equilibrium distribution
            error = []
            for lag in lags:
                if lag == 0:
                    error.append(np.zeros(k-1))
                else:
                    sampler = tmatrix_sampler.TransitionMatrixSampler(self.count_matrix(lag=lag),
                            reversible=self._is_reversible)
                    samples = sampler.sample(n_bootstraps)
                    error.append(np.array([
                        mm.analyze.trajectory.Timescales(
                            T, lag, k=k, sparse=self._is_sparse, rev=self._is_reversible, **kwargs)
                        for T in samples]).std(0))
            return its, np.array(error)
        else:
            return its

    def voronoi(self, stride=1, clusters=None, pbc=None, bins=100, method='full'):
        idx = [np.random.permutation(np.arange(self._base.n_samples[i]))[::stride] for i in range(self._base.n_sets)]
        train = np.concatenate([self._base.data[i][idx[i],:] for i in range(self._base.n_sets)])
        labels = np.concatenate([self.dtraj[i][idx[i]] for i in range(self._base.n_sets)])
        centroids = self._micro.centroids
        if clusters is None:
            clusters = self.metastable_clusters
        if method == 'full':
            return mm.analyze.voronoi.FullVoronoi(train, centroids, clusters=clusters, pbc=pbc, bins=bins)
        elif method == 'classify':
            raise AttributeError('Note yet implemented')
            #return mm.analyze.voronoi.ClassifyVoronoi(train, labels, bins=bins)

# Functions that conditional prepare parameters for functions

def _lags(self, lags):
    # prepare lag as list
    if lags is None:
        lags = [self._base.lag]
    if type(lags) == int:
        lags = [lags]
    return lags


def _n_its(self, n_its):
    # prepare number of implied timescales
    if n_its == -1:
        # for sparse methods must be 1 less than max
        n_its = self._N - 2
    else:
        # first timescale is inifinity
        n_its = n_its + 1
    return n_its

def _GetTransitionMatrix(self, lag, precomputed):
    if precomputed == True:
        try:
            T = self._T
        except:
            T = self.transition_matrix(lag=lag)
    else:
        T = self.transition_matrix(lag=lag)
    return T


def _Timescales(self, lags, k, **kwargs):
    # auto run relaxation timescales from dtraj
    its = []
    for lag in lags:
        if lag == 0:
            its.append(np.zeros(k-1))
        else:
            T = _GetTransitionMatrix(self, lag, False)
            its.append(mm.analyze.trajectory.Timescales(
                T, lag, k=k, sparse=self._is_sparse, rev=self._is_reversible, **kwargs))
    return np.array(its)
