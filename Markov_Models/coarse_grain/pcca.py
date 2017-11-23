import numpy as np
from copy import deepcopy


class PCCA(object):
    def __init__(self, n_states=2, psi=None, ncv=None):
        self.n_states = n_states
        if n_states < 2:
            raise ValueError('''Coarse-graining with PCCA+ requires at least
                                two states for valid decomposition.''')
        self.psi = psi
        self.ncv = ncv

    def coarse_grain(self, model):
        def spread(v):
            return v.max() - v.min()

        if self.psi is None:
            self.psi = model.eigenvectors('right',
                                          k=self.n_states + 1, ncv=self.ncv)
        self.psi = self.psi[:, 1:]

        self.chi = np.zeros(self.psi.shape[0], dtype=int)
        for ma in range(self.n_states - 1):
            v = self.psi[:, ma]
            index = np.argmax([spread(v[self.chi == mi])
                               for mi in range(ma + 1)])
            self.chi[(self.chi == index) & (v > 0)] = ma + 1

        # Build new coarse-grained model
        new_model = deepcopy(model)
        new_model.n_states = self.n_states
        new_model.crisp_membership = self.chi

        S = [np.where(self.chi == ma) for ma in range(self.n_states)]
        new_model._C = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                new_model._C[i, j] = model._C[S[i]][:, S[j]].sum()
        del new_model._T
        _ = new_model.transition_matrix
        return new_model
