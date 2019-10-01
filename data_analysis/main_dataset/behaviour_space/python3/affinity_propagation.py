import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import squareform
import time


if __name__ == "__main__":

    distance_matrix_path = 'D:\\Duncan\\Data\\distance_matrix.npy'
    cluster_labels_path = 'D:\\Duncan\\Data\\cluster_labels_{}.npy'
    cluster_centres_path = 'D:\\Duncan\\Data\\cluster_centres_{}.npy'

    for preference in [1000, 2000, 6000, 8000]:

        print('Clustering preference:', preference)

        print('Importing distance matrix...')
        D = np.load(distance_matrix_path).astype('float32')
        print(D.min(), np.median(D), D.max())
        D = squareform(D)
        print(D.dtype)
        print('done!\n')

        start = time.time()
        clusterer = AffinityPropagation(affinity='precomputed', preference=-preference, copy=False)
        clusterer.fit_predict(-D)
        cluster_labels = clusterer.labels_
        cluster_centres = clusterer.cluster_centers_indices_
        print('Number of clusters:', len(cluster_centres))
        end = time.time()
        print('Time:', (end - start) / 60., '\n')
        np.save(cluster_labels_path.format(preference), cluster_labels)
        np.save(cluster_centres_path.format(preference), cluster_centres)

        del clusterer
        del D
