from paths import paths
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.spatial.distance import squareform


if __name__ == "__main__":

    bout_indices = np.load(paths['bout_indices'])
    exemplars = pd.read_csv(paths['exemplars'], index_col='bout_index')
    exemplars = exemplars[exemplars['clean']]
    exemplar_indices = np.where(np.isin(bout_indices, exemplars.index, assume_unique=True))[0]
    n = len(bout_indices)

    take_indices = []
    for i in exemplar_indices:
        for j in exemplar_indices:
            if j > i:
                take_indices.append(comb(n, 2, exact=True) - comb(n - i, 2, exact=True) + (j - i - 1))
    take_indices = np.array(take_indices)
    assert len(exemplar_indices) == len(squareform(take_indices))

    print 'Opening distance matrix...',
    distance_matrix = np.load(paths['distance_matrix']).astype('float32')
    print 'done!'
    exemplar_matrix = distance_matrix[take_indices]
    del distance_matrix
    exemplar_matrix = squareform(exemplar_matrix)
    np.save(paths['exemplar_distance_matrix'], exemplar_matrix)
