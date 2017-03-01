import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from Markov_Models.src import _voronoi

def ClassifyVoronoi(data, clusters=None, extent=None, bins=100, **kwargs):
    n_features = data.shape[1]
    if extent is None:
        his,ext = np.histogramdd(data, bins=bins)
        extent = []
        for i in range(len(ext)):
            extent.append(ext[i].min())
            extent.append(ext[i].max())
    extent = np.split(np.array(extent), n_features)

    if clusters is None:
        clusters = np.arange(data.shape[0])

    if type(bins) == int:
        bins = [bins for i in range(n_features)]

    q = []
    for i in range(n_features):
        q.append(np.linspace(extent[i][0],extent[i][1],bins[i]))
    Q = np.meshgrid(*q)

    knc = KNeighborsClassifier(n_neighbors=1, **kwargs)
    knc.fit(data, clusters)
    return knc.predict(np.column_stack([Q[i].flatten() for i in range(n_features)])).reshape(*bins)

def FullVoronoi(data, centroids, clusters=None, pbc=None, bins=100):
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

    states = _voronoi.initialize(norm_data, norm_centroids, clusters, pbc)

    if n_features-1 == 1:
        return states.reshape(bins,bins)
    else:
        return states.reshape(n_features-1,bins,bins)
