from datasets.main_dataset import experiment
from eye_convergence_analysis import eye_convergence_directory, convergence_scores_path
from behaviour_analysis.analysis.bouts import BoutData
from behaviour_analysis.miscellaneous import print_heading
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    bouts_path = os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv')
    bouts_df = pd.read_csv(bouts_path, index_col='bout_index', dtype={'ID': str, 'video_code': str})

    convergence_scores = pd.read_csv(convergence_scores_path, dtype={'ID': str})

    frame_rate = 500.
    window = int(0.02 * 500)

    # Import bout data
    bouts = BoutData.from_directory(bouts_df, experiment.subdirs['kinematics'],
                                    check_tail_lengths=False, tail_columns_only=False)

    print_heading('CLASSIFYING BOUTS')
    convergence_states = np.empty((len(bouts_df), 4))
    i = 0
    for idx, fish_info in convergence_scores.iterrows():
        print fish_info.ID
        for bout in bouts.list_bouts(IDs=[fish_info.ID]):
            bout_convergence = np.degrees(bout['right'] - bout['left'])
            convergence_start = bout_convergence[:window].mean()
            convergence_end = bout_convergence[-window:].mean()
            convergence_states[i, :2] = np.array([convergence_start, convergence_end])
            convergence_states[i, 2:] = (np.array([convergence_start, convergence_end]) >= fish_info.threshold)
            i += 1
    assert i == len(convergence_states), 'Incorrect number of bouts!'
    np.save(os.path.join(eye_convergence_directory, 'convergence_states.npy'), convergence_states)

    # Find bout phases
    convergence_states = convergence_states[:, 2:].astype('bool')
    spontaneous = (~convergence_states[:, 0]) & (~convergence_states[:, 1])
    early = (~convergence_states[:, 0]) & (convergence_states[:, 1])
    mid = (convergence_states[:, 0]) & (convergence_states[:, 1])
    late = (convergence_states[:, 0]) & (~convergence_states[:, 1])

    phase_labels = np.column_stack([spontaneous, early, mid, late])
    phase_labels = np.argwhere(phase_labels)[:, 1]
    bouts_df['phase'] = phase_labels
    bouts_df.to_csv(bouts_path)

    # Calculate the proportions of bouts in each cluster belonging to each phase
    n_labels = len(bouts_df['state'].unique())
    phase_counts = np.zeros((n_labels, 4))
    for l, p in zip(bouts_df['state'], bouts_df['phase']):
        phase_counts[l, p] += 1
    normalised_phase_counts = (phase_counts.T / np.sum(phase_counts, axis=1)).T

    np.save(os.path.join(eye_convergence_directory, 'state_convergence_scores.npy'), normalised_phase_counts)
