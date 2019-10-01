from datasets.main_dataset import experiment
from behaviour_analysis.miscellaneous import print_heading
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.analysis.alignment import calculate_distance_matrix
from behaviour_analysis.analysis.embedding import IsomapPrecomputed
from behaviour_analysis.analysis.bouts import BoutData
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform


if __name__ == "__main__":

    ndims = 6

    behaviour_space_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space')
    output_directory = create_folder(behaviour_space_directory, 'six_principal_components')

    # Explained variance
    explained_variance = np.load(os.path.join(behaviour_space_directory, 'explained_variance.npy'))
    print '{} principal components explain:'.format(ndims), np.cumsum(explained_variance)[ndims - 1]

    # Import exemplar bouts
    exemplars = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'exemplars.csv'), index_col='bout_index',
                            dtype=dict(ID=str, video_code=str))
    exemplars = exemplars[exemplars['clean']]
    exemplar_bouts = BoutData.from_directory(exemplars, experiment.subdirs['kinematics'],
                                             check_tail_lengths=False, tail_columns_only=True)

    # Transform bouts
    transformed, pca = exemplar_bouts.transform(whiten=True)
    transformed_bouts = transformed.list_bouts(values=True, ndims=ndims)

    # Calculate distance matrices
    print_heading('CALCULATING DISTANCE MATRIX - NORMAL')
    distance_matrix_1 = calculate_distance_matrix(transformed_bouts, fs=500., flip=False)

    print_heading('CALCULATING DISTANCE MATRIX - FLIPPED')
    distance_matrix_2 = calculate_distance_matrix(transformed_bouts, fs=500., flip=True)

    distance_matrix = np.min([distance_matrix_1, distance_matrix_2], axis=0)
    distance_matrix = squareform(distance_matrix)
    np.save(os.path.join(output_directory, 'distance_matrix.npy'), distance_matrix)

    # Perform embedding
    print_heading('Isomap embedding')
    isomap = IsomapPrecomputed(n_components=20, n_neighbors=5)

    embedding = isomap.fit_transform(distance_matrix)
    kernel_pca_eigenvalues = isomap.kernel_pca_.lambdas_
    reconstruction_errors = isomap.reconstruction_errors()

    np.save(os.path.join(output_directory, 'embedding.npy'), embedding)
    np.save(os.path.join(output_directory, 'eigenvalues.npy'), kernel_pca_eigenvalues)
    np.save(os.path.join(output_directory, 'reconstruction_errors.npy'), reconstruction_errors)
