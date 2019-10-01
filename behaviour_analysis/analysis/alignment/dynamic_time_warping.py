import numpy as np
from scipy.spatial.distance import cdist


def dtw(s, t, bandwidth=0.01, fs=500.):
    # check trajectories have same number of dimensions
    assert s.shape[1] == t.shape[1]
    ndim = s.shape[1]

    # pad ends with zeros
    n = max([len(s), len(t)])
    t0, t1 = np.zeros((n, ndim)), np.zeros((n, ndim))
    t0[:len(s)] = s
    t1[:len(t)] = t

    # calculate bandwidth
    bw = int(bandwidth * fs)

    # initialise distance matrix
    DTW = np.empty((n, n))
    DTW.fill(np.inf)

    # calculate pairwise distances between points on the trajectories
    pairwise_distances = cdist(t0, t1)

    # fill the first row without a cost allowing optimal path to be found starting anywhere within the bandwidth
    DTW[0, :bw] = pairwise_distances[0, 0:bw]

    # main loop of dtw algorithm
    for i in range(1, n):
        for j in range(max(0, i - bw + 1), min(n, i + bw)):
            DTW[i, j] = pairwise_distances[i, j] + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    # return dtw distance
    warping_distance = DTW[-1, -1]

    return warping_distance


def dtw_1d(s, t, bandwidth=0.01, fs=500.):
    # check trajectories have same number of dimensions
    assert s.ndim == t.ndim == 1
    assert len(s) == len(t)
    n = len(s)

    # calculate bandwidth
    bw = int(bandwidth * fs)

    # initialise distance matrix
    DTW = np.empty((n, n))
    DTW.fill(np.inf)

    # fill the first row and first column without a cost
    # allows optimal path to be found starting anywhere within the bandwidth
    DTW[0, :bw] = np.array([np.abs(s[0] - t[j]) for j in range(0, bw)])

    # main loop of dtw algorithm
    for i in range(1, n):
        for j in range(max(0, i - bw + 1), min(n, i + bw)):
            DTW[i, j] = np.abs(s[i] - t[j]) + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    # return dtw distance scaled by the ratio of the bout lengths
    warping_distance = DTW[-1, -1]

    return warping_distance


def fill_row(*bouts, **kwargs):
    s = bouts[0]
    bw = kwargs.get('bw', 0.01)
    fs = kwargs.get('fs', 500.)
    if kwargs.get('flip', False):
        row = np.array([dtw(s, -t, bandwidth=bw, fs=fs) for t in bouts[1:]])
    else:
        row = np.array([dtw(s, t, bandwidth=bw, fs=fs) for t in bouts[1:]])
    return row


def fill_row_min(*bouts, **kwargs):
    s = bouts[0]
    bw = kwargs.get('bw', 0.01)
    fs = kwargs.get('fs', 500.)
    row = np.array([min(dtw(s, t, bandwidth=bw, fs=fs), dtw(s, -t, bandwidth=bw, fs=fs)) for t in bouts[1:]])
    return row


def fill_row_1d(*bouts, **kwargs):
    s = bouts[0]
    bw = kwargs.get('bw', 0.01)
    fs = kwargs.get('fs', 500.)
    row = np.array([dtw_1d(s, t, bandwidth=bw, fs=fs) for t in bouts[1:]])
    return row


def dtw_path(s, t, bandwidth=0.01, fs=500.):
    # check trajectories have same number of dimensions
    assert s.shape[1] == t.shape[1]
    ndim = s.shape[1]

    # pad ends with zeros
    n = max([len(s), len(t)])
    t0, t1 = np.zeros((n, ndim)), np.zeros((n, ndim))
    t0[:len(s)] = s
    t1[:len(t)] = t

    # calculate bandwidth
    bw = int(bandwidth * fs)

    # initialise distance matrix
    DTW = np.empty((n, n))
    DTW.fill(np.inf)

    # calculate pairwise distances between points on the trajectories
    pairwise_distances = cdist(t0, t1)

    # fill the first row without a cost allowing optimal path to be found starting anywhere within the bandwidth
    DTW[0, :bw] = pairwise_distances[0, 0:bw]

    # main loop of dtw algorithm
    for i in range(1, n):
        for j in range(max(0, i - bw + 1), min(n, i + bw)):
            DTW[i, j] = pairwise_distances[i, j] + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    path = [np.array((n - 1, n - 1))]
    while ~np.all(path[-1] == (0, 0)):
        steps = np.array([(-1, 0), (-1, -1), (0, -1)]) + path[-1]
        if np.any(steps < 0):
            idxs = np.ones((3,), dtype='bool')
            idxs[np.where(steps < 0)[0]] = 0
            steps = steps[idxs]
        path.append(steps[np.argmin(DTW[steps[:, 0], steps[:, 1]])])
    path = np.array(path)[::-1]

    return path[:, 0], t1[path[:, 1]], DTW[-1, -1]
