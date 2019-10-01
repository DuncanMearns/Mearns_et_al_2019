import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import expon


def generate_weights(points, bandwidth=40.):
    # Calculate pairwise distances between points
    D = pdist(points)
    # Generates weights
    weight_generator = expon(loc=0, scale=bandwidth)
    weights = weight_generator.pdf(D)
    W = squareform(weights)
    W[np.arange(len(W)), np.arange(len(W))] = weight_generator.pdf(0)
    W = W / W.sum(axis=0)
    return W


def redistribute_transitions(T, W):
    if T.ndim == 2:
        return np.dot(np.dot(W, T), W.T)
    else:
        return np.array([np.dot(np.dot(W, t), W.T) for t in T])
