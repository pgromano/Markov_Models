from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def _KNeighborsClassifier(self, centroids=None, clusters=None, **kwargs):
    '''Classifier implementing the k-nearest neighbors vote.
    Read more in the :ref:`User Guide <classification>`.
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.
    weights : str or callable, optional (default = 'uniform')
        weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric : string or DistanceMetric object (default = 'minkowski')
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.
    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Doesn't affect :meth:`fit` method.
    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.
    .. warning::
       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    '''
    # Determine centroid data
    if centroids is None:
        try:
            c = self.centroids
        except:
            c = self._micro.centroids
    else:
        c = centroids

    # Pass key word arguments
    alg = KNeighborsClassifier(n_neighbors=1)
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

from sklearn.naive_bayes import GaussianNB
def _GaussianNB(self, centroids=None, clusters=None, **kwargs):
    '''Naive Bayes classifier for multinomial models
    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.
    Read more in the :ref:`User Guide <multinomial_naive_bayes>`.
    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.
    intercept_ : property
        Mirrors ``class_log_prior_`` for interpreting MultinomialNB
        as a linear model.
    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.
    coef_ : property
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
        as a linear model.
    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.
    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.
    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    '''
    # Determine centroid data
    if centroids is None:
        try:
            c = self.centroids
        except:
            c = self._micro.centroids
    else:
        c = centroids

    # Pass key word arguments
    alg = GaussianNB()
    for key,val in kwargs.items():
        setattr(alg,key,val)

    # Fit centroid training data to clusters
    if clusters == None:
        # All centroids are unique
        alg.fit(c, np.arange(c.shape[0]))
    else:
        # Each centroid is joined as crisp groups
        alg.fit(c, clusters)

    # Classify all point in simulation
    self.dtraj = [alg.predict(self._base.data[i]) for i in range(self._base.n_sets)]
