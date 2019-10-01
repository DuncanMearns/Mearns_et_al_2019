import numpy as np
from scipy.spatial.distance import pdist, cdist


def energy_statistic(X, Y):
    """Computes the energy statistic between two distributions.

    Parameters
    ----------
    X : np.array (n_samples, n_dims)
        First set of data
    Y : np.array (n_samples, n_dims)
        Second set of data

    Returns
    -------
    T : float
        The energy statistic

    References
    ----------
    https://en.wikipedia.org/wiki/Energy_distance#Energy_statistics
    """

    n, m = float(len(X)), float(len(Y))

    xy = cdist(X, Y)
    xx = pdist(X)
    yy = pdist(Y)

    A = (xy / (n * m)).sum()
    B = 2 * (xx / (n ** 2)).sum()
    C = 2 * (yy / (m ** 2)).sum()

    E = (2 * A) - B - C
    T = ((n * m) / (n + m)) * E

    return T


def compute_e_distance_subset(X, Y=None, n=10000):

    a_indices = np.random.choice(np.arange(len(X)), n, replace=False)
    A = X[a_indices]

    if Y is not None:
        b_indices = np.random.choice(np.arange(len(Y)), n, replace=False)
        B = Y[b_indices]
    else:
        b_indices = np.random.choice(np.arange(len(X)), n, replace=False)
        B = X[b_indices]

    a_dist = pdist(A)
    b_dist = pdist(B)
    ab_dist = cdist(A, B)

    e = (ab_dist.sum() - (0.5 * a_dist.sum()) - (0.5 * b_dist.sum())) / float(n)
    return e
