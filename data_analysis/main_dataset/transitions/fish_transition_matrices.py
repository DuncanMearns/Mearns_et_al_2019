from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.analysis.transitions import fish_transition_matrices, generate_weights, redistribute_transitions
import pandas as pd
import numpy as np
import os


transition_directory = create_folder(experiment.subdirs['analysis'], 'transitions')


if __name__ == "__main__":

    # Import data
    bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index_col='transition_index',
                        dtype={'ID': str, 'video_code': str})
    isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))[:, :3]

    # Compute the transition matrix for each fish
    T = fish_transition_matrices(bouts)
    np.save(os.path.join(transition_directory, 'transition_matrices.npy'), T)
    print T.shape

    # Redistribute transitions
    W = generate_weights(isomap, bandwidth=40.)
    WTW = redistribute_transitions(T, W)
    np.save(os.path.join(transition_directory, 'smoothed_transition_matrices.npy'), WTW)

    # Sum transitions over all fish
    T_all = T.sum(axis=0)
    WTW_all = redistribute_transitions(T_all, W)
    np.save(os.path.join(transition_directory, 'T.npy'), T_all)
    np.save(os.path.join(transition_directory, 'WTW.npy'), WTW_all)
