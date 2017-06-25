import numpy as np

def sample(data, fraction=0.1, shuffle=True):
    # Check fractions are within range
    if fraction == 0 or fraction > 1:
        raise AttributeError('''
        Fraction must be value 0 < f <= 1.''')

    # Prepare data
    if type(data) != list:
        data = [data]
    n_sets = len(data)
    n_samples = []
    for i in range(n_sets):
        n_samples.append(data[i].shape[0])

    stride = int(1/fraction)
    if shuffle is True:
        idx = [np.random.permutation(np.arange(n_samples[i]))[::stride] for i in range(n_sets)]
    else:
        idx = []
        for i in range(n_sets):
            t0 = np.random.choice(np.arange(n_samples[i]-int(n_samples[i]*fraction)))
            idx.append(np.arange(t0,(t0+int(n_samples[i]*fraction))))
    return np.concatenate([data[i][idx[i],:] for i in range(n_sets)])
