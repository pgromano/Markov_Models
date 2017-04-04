from sklearn import metrics
import numpy as np

def Silhouette_Score(data, labels, **kwargs):
    return metrics.silhouette_score(data, labels, **kwargs)

def Silhouette_Samples(data, labels, **kwargs):
    return metrics.silhouette_samples
