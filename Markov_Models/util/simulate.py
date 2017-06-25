from ..analysis.spectral import stationary_distribution as _sd
from scipy.stats import rv_discrete
import numpy as np

def simulate(T, n_samples=1000, n_sets=1, *kwargs):
    pi = _sd(T, *kwargs)
    sampler = rv_discrete(values=(np.arange(T.shape[0]), pi))
    return [sampler.rvs(size=(n_samples)) for i in range(n_sets)]
