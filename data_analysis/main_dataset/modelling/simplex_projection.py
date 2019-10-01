from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.miscellaneous import find_contiguous, print_subheading
import numpy as np
import pandas as pd
import os


modelling_directory = create_folder(experiment.subdirs['analysis'], 'modelling')


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col='transition_index', dtype={'ID': str, 'video_code': str})
    n_states = len(mapped_bouts['module'].unique())

    # ====================
    # Find all bout chains
    # ====================

    print_subheading('Finding bout chains')
    chains = []
    for ID, fish_bouts in mapped_bouts.groupby('ID'):
        print ID
        for video_code, video_bouts in fish_bouts.groupby('video_code'):
            chain = video_bouts['module']
            seqs = find_contiguous(chain.index, minsize=2, stepsize=1)
            for seq in seqs:
                chains.append(chain.loc[seq].values)
    n_chains = len(chains)  # number of unbroken bout chains
    print 'Unbroken bout chains: {}\n'.format(n_chains)

    chains_concatenated = np.concatenate(chains, axis=0)
    state_counts = np.array([(chains_concatenated == s).sum() for s in range(n_states)])
    state_probabilities = (state_counts - 1) / float(state_counts.sum() - 1)

    for state in range(n_states):
        print "simplex projection for state:", state

        state_chains_aligned = np.empty((state_counts[state], 6)) * np.nan
        i = 0
        for chain in chains:
            positions = np.where(chain == state)[0]
            for pos in positions:
                start = max(0, pos - 4)  # start 4 bouts before focal bout
                end = min(pos + 1, len(chain) - 1)  # end one bout after
                state_chains_aligned[i, start - pos + 4 : 5 + end - pos] = chain[start:end+1]
                i += 1
        assert np.all(state_chains_aligned[:, 4] == state)
        assert len(state_chains_aligned) == i

        state_chains_aligned = state_chains_aligned[~np.isnan(state_chains_aligned[:, -1])]

        idxs = np.arange(len(state_chains_aligned))

        prediction_probabilities = np.empty(state_chains_aligned.shape) * np.nan

        for i in range(len(state_chains_aligned)):
            # probability predicted correctly under null
            p0 = state_probabilities[int(state_chains_aligned[i, -1])]
            prediction_probabilities[i, -1] = p0
            # first order markov prediction
            leave_one_out = state_chains_aligned[idxs != i]
            p1 = (leave_one_out[:, -1] == state_chains_aligned[i, -1]).sum() / float(len(leave_one_out))
            prediction_probabilities[i, -2] = p1
            # higher order markov predictions
            for n in (-3, -4, -5, -6):
                if np.isnan(state_chains_aligned[i, n]):
                    continue
                else:
                    leave_one_out = leave_one_out[~np.isnan(leave_one_out[:, n])]
                    match = leave_one_out[np.all((leave_one_out[:, n:-1] == state_chains_aligned[i, n:-1]), axis=1)]
                    if len(match) > 0:
                        p = (match[:, -1] == state_chains_aligned[i, -1]).sum() / float(len(match))
                        prediction_probabilities[i, n] = p
                    else:
                        continue

        sequences_probabilities = np.array([state_chains_aligned, prediction_probabilities])
        np.save(os.path.join(modelling_directory, 'prediction_probabilities_{}.npy'.format(state)), sequences_probabilities)
