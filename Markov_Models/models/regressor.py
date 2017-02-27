from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def _KNeighborsRegressor(self, centroids=None, clusters=None, **kwargs):
    # Determine centroid data
    if centroids is None:
        try:
            c = self.centroids
        except:
            c = self._micro.centroids
    else:
        c = centroids

    # Pass key word arguments
    alg = KNeighborsRegressor(n_neighbors=1)
    for key,val in kwargs.items():
        if key == "n_neighbors" and val != 1:
            warnings.warn('''Centroid assignment is not designed for multiple assignments. Setting n_neighbors = 1''')
        else:
            setattr(alg,key,val)

    # Fit centroid training data to clusters
    if clusters == None:
        # All centroids are unique
        alg.fit(c, np.arange(c.shape[0]))
    else:
        # Each centroid is joined as crisp groups
        alg.fit(c, clusters)
    self.dtraj = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
