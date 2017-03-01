import numpy as np

def initialize(data, centroids, clusters, PBC):

    cdef int i, j, n

    nsamples = data.shape[0]
    nfeatures = data.shape[1]
    ncentroids = centroids.shape[0]

    states = np.zeros(nsamples, dtype=int)
    cost = np.zeros(ncentroids, dtype=float)
    D = np.zeros((nsamples,nfeatures), dtype=float)

    for i from 0 <= i < nsamples:
        for j from 0 <= j < ncentroids:
            for n from 0 <= n < nfeatures:
                if PBC[n] == 1:
                    if abs(data[i,n]-centroids[j,n]) > 0.5:
                        if data[i,n] > centroids[j,n]:
                            D[j,n] = (abs(1-data[i,n])+centroids[j,n])**2
                        elif centroids[j,n] >= data[i,n]:
                            D[j,n] = (abs(1-centroids[j,n])+data[i,n])**2
                    else:
                        D[j,n] = (data[i,n]-centroids[j,n])**2
                elif PBC[n] == 0:
                    D[j,n] = (data[i,n]-centroids[j,n])**2
            cost[j] = np.sqrt(np.sum(D[j,:]))
        n_min = np.argmin(cost)
        states[i] = clusters[n_min[0]]
    return states
