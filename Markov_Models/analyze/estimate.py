#!/usr/bin/python
import Markov_Models as mm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def Metastability(T):
    return np.diagonal(T).sum()

# Probing trajectory Data
def CMatrix(dtraj, lag=1):
    lag = int(lag)
    # Number of simulations
    n_sets = len(dtraj)

    # Length of simulations
    n_samples = [dtraj[i].shape[0] for i in range(n_sets)]
    n_samples = np.array(n_samples).astype(np.int64)
    max_samples = np.amax(n_samples)

    # Number of states
    n_states = np.int64(np.amax(dtraj) + 1)

    # Prepare trajectories as parameter values
    dtraj_store = np.zeros((max_samples, n_sets)).astype(np.int64)
    for i in range(n_sets):
        dtraj_store[:n_samples[i],i] = dtraj[i]

    # Build count matrix
    C = mm.src._estimate.count_matrix(dtraj_store, n_samples, lag, n_states)
    return C

def TMatrix(dtraj, lag=1, rev=True):
    C = CMatrix(dtraj, lag)
    return _estimate_T_matrix(C, rev)

def _estimate_T_matrix(C, rev=True):
    if rev is True:
        Cnr = 0.5*(C+C.T)
        return Cnr/Cnr.sum(1)[:,None]
    else:
        return C/C.sum(1)[:,None]


def Timescales(T, lag, k=1, **kwargs):
    w = mm.analyze.spectral.EigenValues(T, k=k, **kwargs)
    return (-lag / np.log(abs(w[1:])))


def TimescalesSTD(self, dtraj, lags, **kwargs):
    its_full = []
    for i in range(len(dtraj)):
        its = []
        for lag in lags:
            if lag == 0:
                its.append(0)
            else:
                T = TransitionMatrix([dtraj[i]], lag, **kwargs)
                its.append(Timescales(T, lag, **kwargs))
        its_full.append(its)
    return np.std(its_full, axis=0)

def Voronoi(data, centroids, clusters=None, pbc=None, bins=100):
    n_centroids = int(centroids.shape[0])
    n_features = int(data.shape[1])
    n_samples = int(bins**n_features)

    extent = []
    for i in range(n_features):
        extent.append(data[:,i].min())
        extent.append(data[:,i].max())
    extent = np.split(np.array(extent), n_features)

    q = []
    for i in range(n_features):
        q.append(np.linspace(extent[i][0],extent[i][1],bins))
    Q = np.meshgrid(*q)

    data_and_centroids = np.concatenate([centroids, np.column_stack([Q[i].flatten() for i in range(n_features)])])
    norm_data_and_centroids = MinMaxScaler().fit_transform(data_and_centroids)
    norm_data = norm_data_and_centroids[centroids.shape[0]:,:].astype(float, order='F')
    norm_centroids = norm_data_and_centroids[:centroids.shape[0],:].astype(float, order='F')

    if clusters is None:
        clusters = np.arange(1,n_centroids+1)
    else:
        clusters = clusters
        if clusters.min() == 0:
            clusters += 1

    if pbc is None:
        pbc = np.zeros(n_features)
    else:
        pbc = np.array(pbc)

    states = mm.src._voronoi.initialize(norm_data, norm_centroids, clusters, pbc)

    if n_features-1 == 1:
        return states.reshape(bins,bins)
    else:
        return states.reshape(n_features-1,bins,bins)


''' Chapman Kolmogorov Test

References
----------
This test was suggested in [1]_ and described in detail in [2]_.
..  [1]	`F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and T. Weikl:
        Constructing the Full Ensemble of Folding Pathways from Short
        Off-Equilibrium Simulations. Proc. Natl. Acad. Sci. USA, 106,
        19011-19016 (2009)``
..  [2]	`Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D Chodera, C
        Schuette and F Noe. 2011. Markov models of molecular kinetics:
        Generation and validation. J Chem Phys 134: 174105`
'''
# TODO: Prinz et al method
