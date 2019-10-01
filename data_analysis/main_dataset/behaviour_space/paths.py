from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
import os


behaviour_space_directory = create_folder(experiment.subdirs['analysis'], 'behaviour_space')

paths = {}
# Files saved in the experiment directory
paths['bouts'] = os.path.join(experiment.directory, 'bouts.csv')
# Files saved in analysis directory
paths['isomap'] = os.path.join(experiment.subdirs['analysis'], 'isomap.npy')
for fname in ('exemplars', 'mapped_bouts'):
    paths[fname] = os.path.join(experiment.subdirs['analysis'], fname + '.csv')
# Files saved in behaviour space directory
for fname in ('bout_indices', 'eigenfish', 'tail_statistics', 'explained_variance',
              'distance_matrix_normal', 'distance_matrix_flipped', 'distance_matrix',
              'cluster_labels', 'cluster_centres', 'exemplar_distance_matrix',
              'kernel_pca_eigenvalues', 'reconstruction_errors', 'kinematic_features'):
    paths[fname] = os.path.join(behaviour_space_directory, fname + '.npy')
