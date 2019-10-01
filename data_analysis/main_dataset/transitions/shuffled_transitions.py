from behaviour_analysis.analysis.transitions import SVD, fish_transition_matrices, generate_weights, redistribute_transitions
from behaviour_analysis.miscellaneous import Timer
import numpy as np
from datasets.main_dataset import experiment
import os
import pandas as pd
from joblib import Parallel, delayed


def compute_SVD(T, S):
    T_USVs, T_USVa = SVD(T)
    S_USVs, S_USVa = SVD(S)
    Ts = np.diag(T_USVs[1])[:10]
    Ta = np.diag(T_USVa[1])[:20:2]
    Ss = np.diag(S_USVs[1])[:10]
    Sa = np.diag(S_USVa[1])[:20:2]
    return np.array([Ts, Ta, Ss, Sa])


if __name__ == "__main__":

    n_permutations = 1000

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')

    # Import data
    bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index_col='transition_index',
                        dtype={'ID': str, 'video_code': str})
    isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))[:, :3]
    weights = generate_weights(isomap)
    n_fish = len(bouts['ID'].unique())

    timer = Timer()
    timer.start()

    # Compute the average of 100 shuffled transition matrices for each fish
    print 'Generating shuffled matrices...',
    n_shuffles = 100
    S = np.zeros((n_fish, len(isomap), len(isomap)))
    for shuffle in range(n_shuffles):
        S += fish_transition_matrices(bouts, shuffle=True, verbose=False)
    S /= n_shuffles
    S = redistribute_transitions(S, weights)
    np.save(os.path.join(transition_directory, 'shuffled_transition_matrices.npy'), S)
    print timer.convert_time(timer.lap())

    S = np.load(os.path.join(transition_directory, 'shuffled_transition_matrices.npy'))
    T = np.load(os.path.join(transition_directory, 'smoothed_transition_matrices.npy'))

    # Generate permuted matrices
    print 'Generating permutations...',
    S_permuted = np.empty((n_permutations, S.shape[1], S.shape[2]))
    T_permuted = np.empty((n_permutations, T.shape[1], T.shape[2]))
    for permutation in range(n_permutations):
        shuffled_idxs = np.random.permutation(np.arange(n_fish))[:(n_fish + 1) / 2]
        S_permuted[permutation] = S[shuffled_idxs].sum(axis=0)
        T_permuted[permutation] = T[shuffled_idxs].sum(axis=0)
    print timer.convert_time(timer.lap())

    # SVD
    print 'Computing singular-value decompositions...'
    output = Parallel(4)(delayed(compute_SVD)(t, s) for (t, s) in zip(T_permuted, S_permuted))
    output = np.array(output)
    np.save(os.path.join(transition_directory, 'singular_value_permutation.npy'), output)
    print timer.convert_time(timer.stop())
