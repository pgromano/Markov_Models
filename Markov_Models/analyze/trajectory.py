#!/usr/bin/python
import Markov_Models as mm
import numpy as np

def Metastability(T):
    return np.diagonal(T).sum()

# Probing trajectory Data
def CMatrix(dtraj, lag=1):
    # Number of simulations
    n_sets = len(dtraj)

    # Length of simulations
    n_samples = [dtraj[i].shape[0] for i in range(n_sets)]
    max_samples = dtraj[0].shape[0]

    # Number of states
    n_states = np.amax(dtraj) + 1

    # Prepare trajectories as parameter values
    dtraj = np.array(dtraj).T.astype(int, order='F') + 1

    # Build count matrix
    C = np.zeros((n_states, n_states)).astype(int, order='F')
    mm.src._trajectory.c_matrix(C, dtraj, n_samples, lag, n_sets, n_states, max_samples)
    return C.T


def TransitionMatrix(dtraj, lag=1, rev=True, **kwargs):
    # Number of simulations
    n_sets = len(dtraj)

    # Length of simulations
    n_samples = [dtraj[i].shape[0] for i in range(n_sets)]
    max_samples = dtraj[0].shape[0]

    # Number of states
    n_states = np.amax([np.amax(dtraj[i]) for i in range(n_sets)]) + 1

    # Prepare trajectories as parameter values
    dtraj = np.array(dtraj).T.astype(int, order='F') + 1

    # Build transition matrix
    T = np.zeros((n_states, n_states)).astype(float, order='F')
    if rev is True:
        mm.src._trajectory.t_matrix_rev(
            T, dtraj, n_samples, lag, n_sets, n_states, max_samples)
    else:
        mm.src._trajectory.t_matrix_nrev(
            T, dtraj, n_samples, lag, n_sets, n_states, max_samples)
    return T.T


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
