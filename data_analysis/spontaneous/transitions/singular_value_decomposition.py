from datasets.spontaneous import experiment
from behaviour_analysis.analysis.transitions import fish_transition_matrices, generate_weights, redistribute_transitions, SVD
from behaviour_analysis.manage_files import create_folder
import os
import pandas as pd
import numpy as np


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col=0, dtype={'ID': str, 'video_code': str})

    # Re-weighting in isomap space
    isomap = np.load(os.path.join(experiment.parent.subdirs['analysis'], 'isomap.npy'))[:, :3]
    W = generate_weights(isomap)

    # Create paths for saving
    transition_directory = create_folder(experiment.subdirs['analysis'], 'transitions')
    # Compute the transition matrix for each fish
    T = fish_transition_matrices(mapped_bouts, state_col='exemplar', n_states=len(isomap), shuffle=False)
    np.save(os.path.join(transition_directory, 'transition_matrices.npy'), T)
    print T.shape
    # Redistribute transitions
    WTW = redistribute_transitions(T, W)
    np.save(os.path.join(transition_directory, 'smoothed_transition_matrices.npy'), WTW)
    # Sum transitions over all fish
    T_all = T.sum(axis=0)
    WTW_all = redistribute_transitions(T_all, W)
    np.save(os.path.join(transition_directory, 'T.npy'), T_all)
    np.save(os.path.join(transition_directory, 'WTW.npy'), WTW_all)
    # SVD
    USVs, USVa = SVD(WTW_all)
    np.save(os.path.join(transition_directory, 'USVs.npy'), USVs)
    np.save(os.path.join(transition_directory, 'USVa.npy'), USVa)
