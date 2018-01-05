import numpy as np
from sklearn.preprocessing import normalize
from copy import deepcopy


class PCCA(object):
    def __init__(self, n_states=2, psi=None, ncv=None):
        self.n_states = n_states
        if n_states < 2:
            raise ValueError('''Coarse-graining with PCCA+ requires at least
                                two states for valid decomposition.''')
        self.psi = psi
        self.ncv = ncv

    def fit(self, model):
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
        return self

    def transform(self, model):
        # Build new coarse-grained model
        new_model = deepcopy(model)
        new_model.n_states = self.n_states
        new_model.crisp_membership = self.chi

        S = [np.where(self.chi == ma) for ma in range(self.n_states)]
        new_model._T = np.zeros((new_model.n_states, new_model.n_states))
        for i in range(new_model.n_states):
            for j in range(new_model.n_states):
                new_model._T[i, j] = model._T[i, S[j]].sum()
        new_model._T = normalize(new_model._T, norm='l1')
        return new_model

    def fit_transform(self, model):
        self.fit(model)
        return self.transform(model)
