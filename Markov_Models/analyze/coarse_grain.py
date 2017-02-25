import numpy as np
import copy
from msmtools.analysis.api import _pcca_object as _PCCA
import Markov_Models as mm

def PCCA(self, n_macrostates, lag=None):
    # Calculate transition matrix, and if necessary, modify the lag time
    if lag is not None:
        if lag != self._base.lag:
            self._base.lag = lag
            self._micro.lag = lag
        T = self._micro.transition_matrix(lag=lag)
    else:
        T = self._micro.transition_matrix()

    # Coarse grain (PCCA+)
    self._pcca = _PCCA(T, n_macrostates)
    self.memberships = copy.deepcopy(self._pcca.memberships)
    self.metastable_sets = copy.deepcopy(self._pcca.metastable_sets)
    self.metastable_clusters = np.ones(self._micro._N, dtype=int)
    for k, set in enumerate(self.metastable_sets):
        self.metastable_clusters[set] = k + 1

def assign(self):
    n_states = self._base.n_microstates
    n_samples = self._base.n_samples
    sets = np.array(self.metastable_clusters).astype(int, order='F')
    self.dtraj = []
    for i in range(self._base.n_sets):
        dtraj_in = np.array(self._micro.dtraj[i]).astype(int, order='F') + 1
        dtraj_out = np.zeros(n_samples[i]).astype(int, order='F')
        mm.src._trajectory.assignment(dtraj_out, dtraj_in, sets, n_states, n_samples[i])
        self.dtraj.append(dtraj_out)
