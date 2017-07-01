from .base import BaseModel
from .msm import BaseMicroMSM, BaseMacroMSM

import numpy as np
class MSM(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.microstates = BaseMicroMSM(self)
        self.macrostates = BaseMacroMSM(self)

def Markov_Chain(*args, **kwargs):
    if len(args) == 0:
        return ['AffinityPropagation','BayesianGaussianMixture','Birch','GaussianMixture','HMM','KMeans','KMedoids','MeanShift','MiniBatchKMeans']
    else:
        estimator = kwargs.get('estimator', 'KMeans')
        if estimator.lower() == 'affinitypropagation':
            from .cluster import AffinityPropagation
            return AffinityPropagation(*args, **kwargs)
        elif estimator.lower() == 'bayesiangmm':
            from .mixture import BayesianGaussianMixture
            return BayesianGaussianMixture(*args, **kwargs)
        elif estimator.lower() == 'birch':
            from .cluster import Birch
            return Birch(*args, **kwargs)
        elif estimator.lower() == 'gmm':
            from .mixture import GaussianMixture
            return GaussianMixture(*args, **kwargs)
        elif estimator.lower() == 'hmm':
            from .cluster import HMM
            return HMM(*args, **kwargs)
        elif estimator.lower() == 'kmeans':
            from .cluster import KMeans
            return KMeans(*args, **kwargs)
        elif estimator.lower() == 'kmedoids':
            from .cluster import KMedoids
            return KMedoids(*args, **kwargs)
        elif estimator.lower() == 'meanshift':
            from .cluster import MeanShift
            return MeanShift(*args, **kwargs)
        elif estimator.lower() == 'minibatchkmeans':
            from .cluster import MiniBatchKMeans
            return MiniBatchKMeans(*args, **kwargs)
        else:
            raise AttributeError(estimator+' estimator not implemented.')
