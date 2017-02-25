#!/usr/bin/python
import numpy as np
from scipy.linalg import eig, eigh, solve
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse import diags,csr_matrix

def EigenValues(T, k=None, ncv=None, rev=True, sparse=False, **kwargs):
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
            k = min(T.shape[0]-2, 6)
        if ncv is None:
            ncv = min(T.shape[0], max(5*k + 1, 25))
        if rev is True:
            try:
                w = _SparseEigenValuesRev(T, k=k, ncv=ncv)
            except:
                w = _SparseEigenValuesNRev(T, k=k, ncv=ncv)
        else:
            w = _SparseEigenValuesNRev(T, k=k, ncv=ncv)
        return w
    else:
        if rev is True:
            try:
                w = _DenseEigenValuesRev(T, k=k)
            except:
                w = _DenseEigenValuesNRev(T, k=k)
        else:
            w = _DenseEigenValuesNRev(T, k=k)
        if k is None:
            return w
        else:
            return w[:k]

def _DenseEigenValuesRev(T, k=None):
    pi = StationaryDistribution(T, sparse=False)
    Tsym = np.sqrt(pi)
    S = Tsym[:,None] * T / Tsym
    w, R = eigh(S)
    idx = np.argsort(abs(w))[::-1]
    if k is None:
        return w[idx].real
    else:
        return w[idx][:k].real

def _DenseEigenValuesNRev(T, k=None):
    w, R = eig(T)
    idx = np.argsort(abs(w))[::-1]
    if k is None:
        return w[idx].real
    else:
        return w[idx][:k].real

def _SparseEigenValuesRev(T, k=6, ncv=None):
    # Symmetrize Transition Matrix
    pi = StationaryDistribution(T, ncv=ncv, sparse=True)
    Tsym = np.sqrt(pi)

    # Convert T matrix to sparse
    T = csr_matrix(T)
    D = diags(Tsym, 0)
    Dinv = diags(1.0/Tsym, 0)
    S = (D.dot(T)).dot(Dinv)

    # Diagonalize
    w = eigsh(S, k=k, ncv=ncv, which='LM', return_eigenvectors=False)
    idx = np.argsort(abs(w))[::-1]
    return w[idx].real

def _SparseEigenValuesNRev(T, k=6, ncv=None):
    w = eigs(T, k=k, which='LM', ncv=ncv, return_eigenvectors=False)
    idx = np.argsort(abs(w))[::-1]
    return w[idx].real

def EigenVectors(T, k=None, ncv=None, rev=True, left=False, right=True, sparse=False):
    if sparse is True:
        if k is None:
            k = min(T.shape[0]-2, 6)
        if ncv is None:
            ncv = min(T.shape[0], max(5*k + 1, 25))
        if rev is True:
            try:
                w, L, R = _SparseDecompositionRev(T, k=k, ncv=ncv)
            except:
                w, L, R = _SparseDecompositionNRev(T, k=k, ncv=ncv)
        else:
            w, L, R = _SparseDecompositionNRev(T, k=k, ncv=ncv)
    else:
        if rev is True:
            try:
                w, L, R = _DenseDecompositionRev(T, k=k)
            except:
                w, L, R = _DenseDecompositionNRev(T, k=k)
        else:
            w, L, R = _DenseDecompositionNRev(T, k=k)

    if left is True and right is True:
        return L, R
    elif left is True and right is False:
        return L
    elif left is False and right is True:
        return R

def _DenseDecompositionRev(T, k=None):
    # Calculation stationary distribution
    pi = StationaryDistribution(T, sparse=False)

    # Symmetrize Transition matrix
    Tsym = np.sqrt(pi)[:,None]*T/np.sqrt(pi)

    # Diagonalize
    eigvals, eigvecs = eigh(Tsym)

    # Sort
    idx = np.argsort(abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Eigenvalues, and Left and Right eigenvectors
    w = np.diag(eigvals)
    R = eigvecs / np.sqrt(pi)[:,np.newaxis]
    L = eigvecs * np.sqrt(pi)[:,np.newaxis]

    # Enforce positive values
    R00 = R[0,0]
    R[:,0] /= R00

    # Normalize probability vectors
    L[:,0] *= R00

    if k is None:
        return w.real, L.real, R.real
    else:
        return w[:k].real, L[:,:k].real, R[:,:k].real

def _DenseDecompositionNRev(T, k=None):
    # Diagonalize
    w, R = eig(T)

    # Sort
    idx = np.argsort(np.abs(w))[::-1]
    w = w[idx]
    R = R[:, idx]
    L = solve(np.transpose(R), np.eye(T.shape[0]))

    # Normalize
    R[:, 0] = R[:, 0] * np.sum(L[:, 0])
    L[:, 0] = L[:, 0] / np.sum(L[:, 0])
    if k is None:
        return w.real, L.real, R.real
    else:
        return w[:k].real, L[:,:k].real, R[:,:k].real

def _SparseDecompositionRev(T, k=6, ncv=None):
    # Calculation stationary distribution
    pi = StationaryDistribution(T, ncv=ncv, sparse=True)

    # Symmetrize Transition matrix
    T = csr_matrix(T)
    Tsym = np.sqrt(pi)
    Dpi = diags(Tsym, 0)
    Dinv = diags(1.0/Tsym, 0)
    S = (Dpi.dot(T)).dot(Dinv)

    # Eigenvalues, and Left and Right eigenvectors
    ncv = min(T.shape[0], max(4*k + 1, 25))
    eigvals, eigvecs = eigsh(S, k=k, ncv=ncv, which='LM')

    # Sort
    idx = np.argsort(abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Eigenvalues, and Left and Right eigenvectors
    w = np.diag(eigvals)
    R = eigvecs / Tsym[:, np.newaxis]
    L = eigvecs * Tsym[:, np.newaxis]

    # Enforce positive values
    tmp = R[0, 0]
    R[:, 0] = R[:, 0] / tmp

    # Normalize probability vectors
    L[:, 0] = L[:, 0] *  tmp
    return w.real, L.real, R.real

def _SparseDecompositionNRev(T, k=6, ncv=None):
    wR, R = eigs(T, k=k, which='LM', ncv=ncv)
    wL, L = eigs(T.T, k=k, which='LM', ncv=ncv)

    """Sort right eigenvectors"""
    idx = np.argsort(np.abs(wR))[::-1]
    w = wR[idx]
    R = R[:, idx]

    """Sort left eigenvectors"""
    idx = np.argsort(np.abs(wL))[::-1]
    wL = wL[idx]
    L = L[:, idx]

    """l1-normalization of L[:, 0]"""
    L[:, 0] = L[:, 0] / np.sum(L[:, 0])

    """Standard normalization L'R=Id"""
    ov = np.diag(L.T.dot(R))
    R = R / ov[np.newaxis, :]
    return w.real, L.real, R.real

def StationaryDistribution(T, ncv=None, sparse=False):
    if sparse is True:
        w, L = eigs(T.T, k=1, ncv=ncv, which='LR')
        L = L[:, 0].real
        return abs(L)/np.sum(abs(L))
    else:
        w, L = eig(T, left=True, right=False)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        L = L[:, idx]
        return abs(L[:, 0]) / np.sum(abs(L[:, 0]))


def mfpt(T, origin, target):
    def mfpt_solver(T, target):
        dim = T.shape[0]
        A = np.eye(dim) - T
        A[target, :] = 0.0
        A[target, target] = 1.0
        b = np.ones(dim)
        b[target] = 0.0
        return solve(A, b)
    pi = StationaryDistribution(T)

    """Stationary distribution restriced on starting set X"""
    nuX = pi[origin]
    piX = nuX / np.sum(nuX)

    """Mean first-passage time to Y (for all possible starting states)"""
    tY = mfpt_solver(T, target)

    """Mean first-passage time from X to Y"""
    tXY = np.dot(piX, tY[origin])
    return tXY

def error(T):
    N = T.shape[0]
    pi = StationaryDistribution(T)
    E = np.array([[(T[i,j] - T[j,i]*pi[i]/pi[j])**2 for i in range(N)] for j in range(N)])
    return E.sum(0)/(N-1)

def GrahamSchmidt(X, row_vecs=True, norm=True):
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag(
            (X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
