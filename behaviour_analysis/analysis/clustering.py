import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import linkage, fcluster


def affinity_propagation(D, preference='median'):
    assert D.shape[0] == D.shape[1], 'Matrix is not square!'

    if preference in ['minimum', 'min']:
        preference = np.min(-D)
    elif preference in ['maximum', 'max']:
        preference = np.max(-D)
    elif preference in ['median', 'med']:
        preference = np.median(-D)
    else:
        raise ValueError('preference must be: {minimum, maximum, median}')

    clusterer = AffinityPropagation(affinity='precomputed', preference=preference)
    clusterer.fit_predict(-D)
    return clusterer


def hierarchical_clustering(isomap, n_clusters):
    Z = linkage(isomap, 'ward')
    exemplar_cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    exemplar_cluster_labels -= 1
    return exemplar_cluster_labels
