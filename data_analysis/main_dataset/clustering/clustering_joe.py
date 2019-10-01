from datasets.main_dataset import experiment
import os
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import Isomap
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


if __name__ == "__main__":

    isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))

    D = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'exemplar_distance_matrix.npy'))

    USVs = np.load(os.path.join(experiment.subdirs['analysis'], 'transitions', 'USVs.npy'))
    USVa = np.load(os.path.join(experiment.subdirs['analysis'], 'transitions', 'USVa.npy'))

    q, r = np.linalg.qr(np.concatenate([USVs[2, :, 1:3],
                                        USVa[2, :, :2]], axis=1))
    P = squareform(pdist(q))

    # mapper = Isomap(n_neighbors=5, n_components=20)
    # reduced = mapper.fit_transform(P * D)
    reduced = isomap[:, :3]

    Z = hierarchy.linkage(reduced[:, :], method='ward')
    n_clusters = 10
    labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust') - 1
    # labels = KMeans(n_clusters=10).fit_predict(reduced)

    plt.figure()
    plt.scatter(*isomap[:, :2].T, s=10, c=labels, lw=0, cmap='nipy_spectral')
    plt.show()

    # num clusters
    clusterrange = range(6, 14)
    sil_scores = [metrics.silhouette_score(reduced, hierarchy.fcluster(Z, n_clusters, criterion='maxclust') - 1,
                                           metric='euclidean') for n_clusters in clusterrange]
    # sil_scores = [metrics.silhouette_score(reduced, KMeans(n_clusters=n_clusters).fit_predict(reduced),
    #                                        metric='euclidean') for n_clusters in clusterrange]
    plt.figure()
    plt.plot(clusterrange, sil_scores)
    plt.show()

    # np.save(os.path.join(experiment.subdirs['analysis'], 'clustering', 'cluster_labels'), labels)
