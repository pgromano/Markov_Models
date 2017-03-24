import numpy as np
import copy, warnings
from msmtools.analysis.api import _pcca_object as _PCCA
from . import src

def PCCA(self, n_macrostates, lag=None):
    # Function to convert microstate trajectory to macrostate assignments
    def _assign_macrostates(dtraj, sets):
        return np.asarray(src._assignment.crisp_assignment(dtraj, sets))

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
    self.metastable_assignment = np.ones(self._micro._N, dtype=int)
    for k, set in enumerate(self.metastable_sets):
        self.metastable_assignment[set] = k + 1

    # Build discrete macrostate trajectories
    self.dtraj = [_assign_macrostates(self._micro.dtraj[i],
                    self.metastable_assignment)
                    for i in range(self._base.n_sets)]

# TODO: Implement BACE
def BACE(self, n_macrostates):
    pass

from hmmlearn.hmm import GaussianHMM as _HMM
def HMM(self, n_macrostates):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        hmm = _HMM(n_components=n_macrostates).fit(self._micro.centroids)
        self.dtraj = [hmm.predict(self._base.data[i]) for i in range(self._base.n_sets)]
        self.metastable_assignment = hmm.predict(self._micro.centroids)

from sklearn.mixture import GaussianMixture as _GMM
def GMM(self, n_macrostates):
    gmm = _GMM(n_components=n_macrostates).fit(self._micro.centroids)
    self.dtraj = [gmm.predict(self._base.data[i]) for i in range(self._base.n_sets)]
    self.metastable_assignment = gmm.predict(self._micro.centroids)
