import numpy as np
import copy
from msmtools.analysis.api import _pcca_object as _PCCA
from . import src

def PCCA(self, n_macrostates, lag=None):
    # Calculate transition matrix, and if necessary, modify the lag time
    if lag is None:
        T = self._micro._T
    else:
        T = self._micro._transition_matrix(lag=lag)

    # Coarse grain (PCCA+)
    self._pcca = _PCCA(T, n_macrostates)
    self.memberships = copy.deepcopy(self._pcca.memberships)
    self.metastable_sets = copy.deepcopy(self._pcca.metastable_sets)
    self.metastable_clusters = np.ones(self._micro._N, dtype=int)
    for k, set in enumerate(self.metastable_sets):
        self.metastable_clusters[set] = k + 1
    self.dtraj = [_assign_macrostates(self._micro.dtraj[i],
                    self.metastable_clusters)
                    for i in range(self._base.n_sets)]

# TODO: Implement BACE or alternative coarse graining methods???

def _assign_macrostates(dtraj, sets):
    return src._assignment.crisp_assignment(dtraj, sets)
