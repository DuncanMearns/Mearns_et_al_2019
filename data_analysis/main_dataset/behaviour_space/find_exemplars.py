from paths import paths
from behaviour_analysis.analysis.clustering import affinity_propagation
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd


if __name__ == "__main__":

    cluster_size_threshold = 3

    distance_matrix_n = np.load(paths['distance_matrix_normal']).astype('float32')
    distance_matrix_f = np.load(paths['distance_matrix_flipped']).astype('float32')

    distance_matrix = np.min([distance_matrix_n, distance_matrix_f], axis=0)
    del distance_matrix_n
    del distance_matrix_f
    np.save(paths['distance_matrix'], distance_matrix)

    distance_matrix = squareform(distance_matrix)
    clusterer = affinity_propagation(distance_matrix)
    cluster_labels = clusterer.labels_
    cluster_centres = clusterer.cluster_centers_indices_

    n_clusters = len(np.unique(cluster_labels))
    print 'Number of affinity propagation clusters: {}'.format(n_clusters)

    del distance_matrix

    cluster_sizes = np.array([np.sum(cluster_labels == l) for l in np.unique(cluster_labels)])
    big_clusters = np.where(cluster_sizes >= cluster_size_threshold)[0]
    exemplar_indices = cluster_centres[big_clusters]

    bout_indices = np.load(paths['bout_indices'])
    exemplar_bout_indices = bout_indices[exemplar_indices]
    bouts_df = pd.read_csv(paths['bouts'], dtype={'ID': str, 'video_code': str})
    exemplars = bouts_df.loc[exemplar_bout_indices]
    exemplars['cluster'] = big_clusters
    exemplars['clean'] = np.NaN
    exemplars = exemplars.reset_index()
    exemplars = exemplars.rename(columns={'index': 'bout_index'})
    exemplars = exemplars[['ID', 'video_code', 'start', 'end', 'bout_index', 'clean']]

    np.save(paths['cluster_labels'], cluster_labels)
    np.save(paths['cluster_centres'], cluster_centres)
    exemplars.to_csv(paths['exemplars'], index=False)
