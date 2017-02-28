from sklearn import metrics
import numpy as np

def silhouette_score(data, labels, **kwargs):
    return metrics.silhouette_score(data, labels, **kwargs)

def silhouette_samples(data, labels, **kwargs):
    return metrics.silhouette_samples
