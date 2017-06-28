from . import src
from .spectral import stationary_distribution as _sd

import numpy as np
import copy, warnings
from msmtools.analysis.api import _pcca_object as _PCCA
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier

def PCCA(self, n_macrostates, lag=None):
    # Function to convert microstate trajectory to macrostate assignments
    def _assign_macrostates(labeled_data, sets):
        return np.asarray(src._assignment.crisp_assignment(labeled_data, sets))

    # Calculate transition matrix, and if necessary, modify the lag time
    if lag is None:
        T = self._micro._T
    else:
        T = self._micro._transition_matrix(lag=lag)

    # Coarse grain (PCCA+)
    self._pcca = _PCCA(T, n_macrostates)

    # Inherit PCCA assignments and memberships
    self.memberships = copy.deepcopy(self._pcca.memberships)
    self.metastable_sets = copy.deepcopy(self._pcca.metastable_sets)
    self.metastable_labels = np.ones(self._micro._N, dtype=int)
    for k,state in enumerate(self.metastable_sets):
        self.metastable_labels[state] = k

    # Build discrete macrostate trajectories
    self.labels = [_assign_macrostates(self._micro.labels[i],
                    self.metastable_labels)
                    for i in range(self._base.n_sets)]

    M = self.memberships
    pi = _sd(T, ncv=None, sparse=self._is_sparse)

    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, T), M)
    T_coarse = np.dot(W, A)

    pi_coarse = np.dot(M.T, pi)
    X = np.dot(np.diag(pi_coarse), T_coarse)
    self._T = X / X.sum(axis=1)[:, None]

# TODO: Implement BACE
def BACE(self, n_macrostates):
    pass

from hmmlearn.hmm import GaussianHMM as _HMM
def HMM(self, n_macrostates):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        hmm = _HMM(n_components=n_macrostates).fit(self._micro.centroids)
        self.labels = [hmm.predict(self._base.data[i]) for i in range(self._base.n_sets)]
        self.metastable_labels = hmm.predict(self._micro.centroids)
        self.metastable_sets = []
        for i in range(self._N):
            self.metastable_sets.append(np.where(self.metastable_labels == i)[0])

from sklearn.mixture import GaussianMixture as _GMM
def GMM(self, n_macrostates):
    gmm = _GMM(n_components=n_macrostates).fit(self._micro.centroids)
    self.labels = [gmm.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    self.metastable_labels = gmm.predict(self._micro.centroids)
    self.metastable_covariances = gmm.covariances_
    self.aic = gmm.aic
    self.bic = gmm.bic
    self.metastable_sets = []
    for i in range(self._N):
        self.metastable_sets.append(np.where(self.metastable_labels == i)[0])

from sklearn.cluster import AgglomerativeClustering as _HC
def HC(self, n_macrostates):
    scaler = _MinMaxScaler(feature_range=(0,1)).fit(self._micro.centroids)
    hc = _HC(n_clusters=n_macrostates).fit(scaler.transform(self._micro.centroids))
    clf = _KNeighborsClassifier(n_neighbors=1).fit(scaler.transform(self._micro.centroids),hc.labels_)

    self.labels = [clf.predict(scaler.transform(self._base.data[i])) for i in range(self._base.n_sets)]
    self.metastable_labels = hc.labels_
    self.metastable_sets = []
    for i in range(self._N):
        self.metastable_sets.append(np.where(hc.labels_ == i)[0])
