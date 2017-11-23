import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs


def values(T, sparse, k=None, ncv=False):
    r'''Compute eigenvalues of transition matrix.
    Parameters
    ---------_
    T : (n, n) numpy.ndarray
        Transition matrix (row stochastic)
    k : int, optional
        The number of eigenvalues and eigenvectors desired. k must be smaller
        than N. It is not possible to compute all eigenvectors of a matrix.
    ncv : int, option
        The number of Lanczos vectors generated ncv must be greater than k; it
        is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    Returns
    -------
    w : (k,) numpy.ndarray
        Array of k eigenvalues.
    '''
    if sparse is True:
        if k is None:
            k = min(T.shape[0] - 2, 6)
        if ncv is None:
            ncv = min(T.shape[0], max(5 * k + 1, 25))

        # Solve Eigenvalues
        w, R = eigs(T, k=k, ncv=ncv)
    else:
        # Solve Eigenvalues
        w, R = eig(T)

        # Sort Eigenvalues
        index = np.argsort(abs(w))[::-1]
        w = w[index]

        # Subset of k Eigenvalues
        if k is not None:
            w = w[:k]
    return w.real


def vectors(T, method, sparse, k=None, ncv=False):
    r'''
    Compute eigenvalues of transition matrix.
    Parameters
    ---------
    T : (n, n) numpy.ndarray
        Transition matrix (row stochastic)
    k : int, optional
        The number of eigenvalues and eigenvectors desired. k must be smaller
        than N. It is not possible to compute all eigenvectors of a matrix.
    ncv : int, option
        The number of Lanczos vectors generated ncv must be greater than k; it
        is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    Returns
    -------
    w : (k,) numpy.ndarray
        Array of k eigenvalues.
    L : (k, k) numpy.ndarray
        Matrix of left eigenvectors.
    R : (k, k) numpy.ndarray
        Matrix of right eigenvectors.
    '''
    if sparse is True:
        if k is None:
            k = min(T.shape[0] - 2, 6)
        if ncv is None:
            ncv = min(T.shape[0], max(5 * k + 1, 25))

        # Sparse Spectral Decomposition
        wR, R = eigs(T, k=k, which='LM', ncv=ncv)
        wL, L = eigs(T.T, k=k, which='LM', ncv=ncv)
        wL, wR, L, R = wL.real, wR.real, L.real, R.real

        # Sort Right Eigenvectors
        index = np.argsort(np.abs(wR))[::-1]
        w = wR[index]
        R = R[:, index]

        # Sort Left Eigenvectors
        index = np.argsort(np.abs(wL))[::-1]
        wL = wL[index]
        L = L[:, index]
    else:
        # Dense Spectral Decomposition
        w, L, R = eig(T, left=True, right=True)
        w, L, R = w.real, L.real, R.real

        # Sort Eigenvalues and Vectors
        index = np.argsort(abs(w))[::-1]
        w = w[index]
        R = R[:, index]
        L = L[:, index]

        # Reduce Eigen System to Size k
        if k is not None:
            w, L, R = w[:k], L[:, :k], R[:, :k]

    # Normalize the Stationary Distribution
    L[:, 0] = L[:, 0] / np.sum(L[:, 0])

    # Standard normalization L'R=Id
    # norm = np.diag(np.dot(np.transpose(L), R))
    # L[:, 1:] = L[:, 1:] / norm[1:]
    # R = R / norm

    for i in range(1, L.shape[1]):
        L[:, i] = L[:, i] / np.sqrt(np.dot(L[:, i], L[:, i] / L[:, 0]))

    for i in range(R.shape[1]):
        R[:, i] = R[:, i] / np.dot(L[:, i], R[:, i])

    if method == 'both':
        return L, R
    elif method == 'left':
        return L
    elif method == 'right':
        return R
