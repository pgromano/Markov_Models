import numpy as np
from scipy.linalg import eig, solve
from scipy.sparse.linalg import eigs


__all__ = ['distribution', 'mfpt', 'sample', 'simulate', 'timescales']


def distribution(T, ncv=None, sparse=False):
    if sparse is True:
        w, L = eigs(T.T, k=1, ncv=ncv, which='LR')
        L = L[:, 0].real
        return abs(L) / np.sum(abs(L))
    else:
        w, L = eig(T, left=True, right=False)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        L = L[:, idx]
        return abs(L[:, 0]) / np.sum(abs(L[:, 0]))


def mfpt(T, origin, target=None, sparse=False):
    if target is None:
        n_states = T.shape[0]

        # Define General Population Matrix
        A = np.eye(n_states) - T
        A[origin, :] = 0.0
        A[origin, origin] = 1.0

        # Define Target Matrix
        b = np.ones(n_states)
        b[origin] = 0.0
        return solve(A, b)
    else:
        # Stationary distribution restriced on starting set A
        A = distribution(T)[origin]
        A = A / np.sum(A)

        # Mean first-passage time to B (for all possible starting states)
        B = mfpt(T, target)

        # Mean first-passage time from A to B
        return np.dot(A, B[origin])


def sample(pi, n_samples, random_state):
    np.random.seed(random_state)
    return np.random.choice(np.arange(len(pi)), p=pi, size=n_samples)


def timescales(self, **kwargs):
    t = -self.lag / np.log(abs(self.eigenvalues(**kwargs)[1:]))
    return np.append(np.inf, t)
