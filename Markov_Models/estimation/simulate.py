from . import _simulate
import numpy as np

def from_matrix(T, n_samples, n0, random_state):
    if n_samples is None:
        n_samples = 1

    if isinstance(n_samples, int):
        return np.asarray(_simulate.simulate(T, n_samples,
                                             n0, 1, random_state))
    else:
        if len(n_samples) == 1:
            X = np.asarray(_simulate.simulate(
                           T, np.int(np.product(n_samples)),
                           n0, 1, random_state))
        else:
            X = np.asarray(_simulate.simulate(
                           T, np.int(np.product(n_samples)),
                           n0, n_samples[0], random_state))
    return X.reshape(n_samples)

def from_dict(T, n_samples, n_order, n0, random_state):
    dtype = type(list(T.keys())[0][0])

    # Set seed and initialize array
    np.random.seed(random_state)
    X = np.empty(n_samples, dtype=dtype)

    # Initialize simulation
    if n0 is None:
        n_init = len(T.keys())
        n0 = list(T.keys())[np.random.choice(range(n_init))]
    X[:n_order] = n0

    # Simulate
    for n in range(n_order, n_samples):
        prev_step = tuple(X[n - n_order:n])
        keys = list(T[prev_step].keys())
        vals = list(T[prev_step].values())
        X[n] = np.random.choice(keys, size=1, p=vals)[0]
    return X
