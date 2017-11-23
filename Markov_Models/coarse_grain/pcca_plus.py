from ..utils.validation import check_equilibrium
import numpy as np
from scipy.linalg import inv, pinv, norm
from scipy.optimize import basinhopping, fmin
from copy import deepcopy


class PCCAPlus(object):
    def __init__(self, n_states=2, pi=None, psi=None,
                 ncv=None, optimize=True,
                 objective='crisp_metastability'):
        '''
        Parameters:
        -----------
        n_states: int
            Number of states to coarse-grain the model.
        objective_function: str
            Objective functions to minimize the PCCA+ solution. Can be one of
            the following: crispness, crisp_metastability, fuzzy_metastability.
            Additionally, a function can be passed directly.
        '''
        if n_states < 2:
            raise ValueError('''Coarse-graining with PCCA+ requires at least
                                two states for valid decomposition.''')
        self.n_states = n_states
        self.objective = objective
        self.optimize = optimize
        self.ncv = ncv
        self.psi = psi
        self.pi = pi

    def coarse_grain(self, model):
        if self.n_states > model._T.shape[0] - 1:
            raise ValueError('''Number of states to coarse-grain must be one
                                less than states in initial model.''')

        # Solve or check left (pi) and right (psi) eigenvectors
        if self.psi is None:
            if self.pi is None:
                mu, psi = model.eigenvectors('both', k=self.n_states,
                                             ncv=self.ncv)
                self.pi = mu[:, 0]
                self.psi = psi
            else:
                self.pi = check_equilibrium(self.pi, model._T.shape[0])
                self.psi = model.eigenvectors('right', k=self.n_states,
                                              ncv=self.ncv)
        else:
            assert self.psi.shape[0] == model._T.shape[0], 'Invalid right eigenvectors'
            assert self.psi.shape[1] == self.n_states, 'Invalid number of vectors'
            if self.pi is None:
                self.pi = model.equilibrium
            else:
                self.pi = check_equilibrium(self.pi, model._T.shape[0])

        # Get transition matrix to coarsen
        self.T = model.transition_matrix

        # Guess A-matrix
        index = _index_search(psi)
        A = self.psi[index, :]
        try:
            A = inv(A)
        except:
            A = pinv(A)
        A = _fill_A(A, self.psi)

        # Optimize A Matrix
        if self.optimize:
            A = self._optimize_A(A)
        self.A = _fill_A(A, self.psi)

        # Build new coarse-grained model
        new_model = deepcopy(model)
        new_model.n_states = self.n_states
        new_model.fuzzy_membership = self.chi = np.dot(self.psi, self.A)
        new_model.crisp_membership = np.argmax(self.chi, axis=1)

        # Estimate coarse-grained transition matrix
        w = inv(np.dot(self.chi.T, self.chi))
        a = np.dot(np.dot(self.chi.T, model._T), self.chi)
        T_coarse = np.dot(w, a)

        pi_coarse = np.dot(self.chi.T, self.pi)
        X_coarse = np.dot(np.diag(pi_coarse), T_coarse)
        T_coarse = X_coarse / X_coarse.sum(axis=1)[:, None]
        new_model._T = T_coarse
        return new_model

    def _optimize_A(self, A):
        def f(alpha):
            if self.objective == 'crispness':
                return -1 * _crispness(alpha, self.T, self.pi, self.psi)
            elif self.objective == 'roeblitz':
                return -1 * _roeblitz(alpha, self.T, self.pi, self.psi)
            elif self.objective == 'crisp_metastability':
                return -1 * _crisp_metastability(alpha, self.T, self.pi, self.psi)
            elif self.objective == 'fuzzy_metastability':
                return -1 * _fuzzy_metastability(alpha, self.T, self.pi, self.psi)

        shape = A.shape
        alpha = A.ravel()
        alpha = basinhopping(f, alpha, niter_success=1000)['x']
        alpha = fmin(f, alpha,
                     xtol=1e-4, ftol=1e-4,
                     maxfun=1e4, maxiter=1e4,
                     disp=False, full_output=False)

        if np.isneginf(f(alpha)):
            raise ValueError('Minimization has failed to find good solution')
        return alpha.reshape(shape)


def _index_search(psi):
    n_micro, n_macro = psi.shape
    index = np.zeros(n_macro, dtype=int)

    # Find the outermost point in eigenvector space
    index[0] = np.argmax([norm(psi[mi]) for mi in range(n_micro)])

    # Change orthogonal reference frame with outermost point the origin
    ortho = psi - psi[index[0], np.newaxis]

    # Gram-Schmidt orthogonalization
    for ma in range(1, n_macro):
        temp = ortho[index[ma - 1]].copy()
        for mi in range(n_micro):
            ortho[mi] -= temp * np.dot(ortho[mi], temp)
        dist = np.array([norm(ortho[mi]) for mi in range(n_micro)])
        index[ma] = np.argmax(dist)
        ortho /= dist.max()
    return index


def _fill_A(A, psi):
    n_micro, n_macro = psi.shape
    A = A.copy()

    A[1:, 0] = -1 * A[1:, 1:].sum(1)
    A[0] = -1 * np.dot(psi[:, 1:].real, A[1:]).min(0)
    return A / A[0].sum()


def _crispness(alpha, T, pi, psi):
    '''
    Parameters:
    -----------
    A: numpy.ndarray
        The assignment (or transformation) matrix which is the output of the
        constrained optimization problem.
    psi: numpy.ndarray (n_micro, n_macro)
        Right eigenvectors of the transition matrix. This should contain the
        second through n_macro eigenvectors. The first eigenvector should be
        excluded as these are normalized to one.
    Returns:
    --------
    objective: float
        The objective function
    References:
    -----------
    ..  Kube S.,  "Statistical Error Estimation and Grid-free
    Hierarchical Refinement in Conformation Dynamics," Doctoral Thesis.
    2008
    '''
    n_micro, n_macro = psi.shape
    A = alpha.reshape(n_macro, n_macro)
    A = _fill_A(A, psi)

    lhs = 1 - A[0, 1:].sum()
    rhs = np.dot(psi[:, 1:], A[1:, 0])
    rhs = -rhs.min()

    if abs(lhs - rhs) > 1e-8:
        return -np.inf
    score = np.trace(np.dot(np.diag(1 / A[0]), np.dot(A.T, A)))
    return score


def _crisp_metastability(alpha, T, pi, psi):
    n_micro, n_macro = psi.shape
    A = alpha.reshape(n_macro, n_macro)
    A = _fill_A(A, psi)

    fuzzy_membership = np.dot(psi, A)
    crisp_membership = np.argmax(fuzzy_membership, axis=1)
    chi = 0.0 * fuzzy_membership
    chi[np.arange(n_micro), crisp_membership] = 1

    lhs = 1 - A[0, 1:].sum()
    rhs = np.dot(psi[:, 1:], A[1:, 0])
    rhs = -rhs.min()
    if abs(lhs - rhs) > 1e-8:
        return -np.inf

    score = 0.0
    for ma in range(n_macro):
        score += np.dot(T.dot(chi[:, ma]),
                        pi * chi[:, ma]) / np.dot(chi[:, ma], pi)
    return score


def _fuzzy_metastability(alpha, T, pi, psi):
    n_micro, n_macro = psi.shape
    A = alpha.reshape(n_macro, n_macro)
    A = _fill_A(A, psi)
    chi = np.dot(psi, A)

    lhs = 1 - A[0, 1:].sum()
    rhs = np.dot(psi[:, 1:], A[1:, 0])
    rhs = -rhs.min()
    if abs(lhs - rhs) > 1e-8:
        return -np.inf

    score = 0.0
    for ma in range(n_macro):
        score += np.dot(T.dot(chi[:, ma]),
                        pi * chi[:, ma]) / np.dot(chi[:, ma], pi)
    return score


def _roeblitz(alpha, T, pi, psi):
    n_micro, n_macro = psi.shape
    A = alpha.reshape(n_macro, n_macro)
    A = _fill_A(A, psi)

    score = 0
    for i in range(n_macro):
        for j in range(n_macro):
            score += A[j, i] ** 2 / A[0, i]
    return score
