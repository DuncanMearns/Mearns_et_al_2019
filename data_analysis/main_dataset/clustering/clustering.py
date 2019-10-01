from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
import os
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import Isomap
from sklearn import metrics
import pandas as pd


if __name__ == "__main__":

    clustering_directory = create_folder(experiment.subdirs['analysis'], 'clustering')

    isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))
    exemplars = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'exemplars.csv'),
                            dtype={'ID': str, 'video_code': str})
    embedded_idxs = exemplars[exemplars['clean']].index

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col='bout_index',
                               dtype={'ID': str, 'video_code': str})

    D = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'exemplar_distance_matrix.npy'))

    USVs = np.load(os.path.join(experiment.subdirs['analysis'], 'transitions', 'USVs.npy'))
    USVa = np.load(os.path.join(experiment.subdirs['analysis'], 'transitions', 'USVa.npy'))

    q, r = np.linalg.qr(np.concatenate([USVs[2, :, 1:3],
                                        USVa[2, :, :2]], axis=1))
    P = squareform(pdist(q))
    np.save(os.path.join(clustering_directory, 'transition_distances.npy'), P)

    mapper = Isomap(n_neighbors=5, n_components=20)
    reduced = mapper.fit_transform(P * D)

    Z = hierarchy.linkage(reduced[:, :], method='ward')
    n_clusters = 7
    labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust') - 1

    # num clusters
    clusterrange = range(2, 14)
    sil_scores = [metrics.silhouette_score(reduced, hierarchy.fcluster(Z, n_clusters, criterion='maxclust') - 1,
                                           metric='euclidean') for n_clusters in clusterrange]
    sil_scores = np.array(sil_scores)
    np.save(os.path.join(clustering_directory, 'silhouette_scores.npy'), sil_scores)

    exemplars['module'] = -1
    exemplars.loc[embedded_idxs, 'module'] = labels
    exemplars.to_csv(os.path.join(experiment.subdirs['analysis'], 'exemplars.csv'), index=False)

    mapped_bouts['module'] = exemplars.loc[embedded_idxs, 'module'].values[mapped_bouts['state'].values]
    mapped_bouts.to_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index=True)

    np.save(os.path.join(clustering_directory, 'hybrid_space.npy'), reduced)
