from paths import paths
from behaviour_analysis.analysis.embedding import IsomapPrecomputed
import numpy as np
from scipy.spatial.distance import squareform


if __name__ == "__main__":

    D = squareform(np.load(paths['capture_strike_distance_matrix']))
    isomap = IsomapPrecomputed(n_neighbors=5, n_components=2)
    isomapped_strikes = isomap.fit_transform(D)
    isomapped_strikes *= (-1, 1)
    np.save(paths['isomapped_strikes'], isomapped_strikes)
