from datasets.main_dataset import experiment
from behaviour_analysis.analysis.embedding import IsomapPrecomputed
import os
import numpy as np
from scipy.spatial.distance import squareform


if __name__ == "__main__":

    behaviour_space_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space')
    embedding_directory = os.path.join(behaviour_space_directory, 'embedding')

    D = np.load(os.path.join(behaviour_space_directory, 'distance_matrix.npy')).astype('float32')
    D = squareform(D)
    print D.shape

    embedding = IsomapPrecomputed(n_neighbors=20, n_components=2).fit_transform(D)
    np.save(os.path.join(embedding_directory, 'isomap_all.npy'), embedding)
