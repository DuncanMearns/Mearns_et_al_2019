from datasets.main_dataset import experiment
from behaviour_analysis.analysis.embedding import IsomapPrecomputed
import os
import numpy as np
from scipy.spatial.distance import squareform
from scipy.special import comb

import pandas as pd


if __name__ == "__main__":

    preferences = (400, 1000, 2000, 4000, 6000)

    behaviour_space_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space')
    embedding_directory = os.path.join(behaviour_space_directory, 'embeddings')

    print 'Opening distance matrix...',
    distance_matrix = np.load(os.path.join(behaviour_space_directory, 'distance_matrix.npy')).astype('float32')
    print 'done!'

    Ds = []

    print 'Generating exemplar matrices'
    for preference in preferences:
        print preference
        cluster_centres = np.load(os.path.join(embedding_directory, 'cluster_centres_{}.npy'.format(preference)))
        cluster_labels = np.load(os.path.join(embedding_directory, 'cluster_labels_{}.npy'.format(preference)))
        n = len(cluster_labels)
        assert len(cluster_centres) == (cluster_labels.max() + 1)

        cluster_sizes = np.array([(cluster_labels == l).sum() for l in range(len(cluster_centres))])
        cluster_centres = cluster_centres[cluster_sizes >= 3]

        take_indices = []
        for i in cluster_centres:
            for j in cluster_centres:
                if j > i:
                    take_indices.append(comb(n, 2, exact=True) - comb(n - i, 2, exact=True) + (j - i - 1))
        take_indices = np.array(take_indices)
        assert len(cluster_centres) == len(squareform(take_indices))

        Ds.append(distance_matrix[take_indices])

    del distance_matrix

    eigenvalues = []
    reconerrors = []

    np.random.seed(2019)
    print 'Generating embeddings'
    for pref, distances in zip(preferences, Ds):
        print pref
        D = squareform(distances)
        isomap = IsomapPrecomputed(n_components=10)
        embedding = isomap.fit_transform(D)
        eig = isomap.kernel_pca_.lambdas_
        err = isomap.reconstruction_errors()
        np.save(os.path.join(embedding_directory, 'isomap_{}.npy'.format(pref)), embedding)
        eigenvalues.append(eig)
        reconerrors.append(err)
    eigenvalues = np.array(eigenvalues)
    reconerrors = np.array(reconerrors)
    np.save(os.path.join(embedding_directory, 'eigenvalues.npy'), eigenvalues)
    np.save(os.path.join(embedding_directory, 'reconstruction_errors.npy'), reconerrors)
