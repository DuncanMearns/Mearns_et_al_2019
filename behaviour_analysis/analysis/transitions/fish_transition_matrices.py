from .transition_matrix import transition_matrix
from ...miscellaneous import Timer
import numpy as np


def fish_transition_matrices(bouts, state_col='state', n_states=None, shuffle=False, verbose=True):
    if n_states is None:
        n_states = bouts[state_col].max() + 1
    fish_IDs = bouts['ID'].unique()
    n_fish = len(fish_IDs)
    # Start timer
    timer = Timer()
    timer.start()
    if verbose:
        if shuffle:
            print 'Computing shuffled transition matrices...',
        else:
            print 'Computing transition matrices...',
    # Matrices
    T = np.zeros((n_fish, n_states, n_states))
    # Iterate through fish
    bouts_by_fish = bouts.groupby('ID')
    for idx, ID in enumerate(fish_IDs):
        fish_bouts = bouts_by_fish.get_group(ID)
        if shuffle:
            fish_bouts[state_col] = np.random.permutation(fish_bouts[state_col].values)
        fish_T = np.zeros((n_states, n_states))
        for video_code, video_bouts in fish_bouts.groupby('video_code'):
            states = video_bouts[state_col]
            video_T = transition_matrix(states.values, states.index, n_states)
            fish_T += video_T
        T[idx] = fish_T
    # Finishing up
    if verbose:
        print 'done!', timer.convert_time(timer.stop())
    return T
