from datasets.lensectomy import experiment
from behaviour_analysis.analysis import BoutData
from behaviour_analysis.miscellaneous import print_heading
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Open bouts
    bouts_path = os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv')
    bouts_df = pd.read_csv(bouts_path, index_col=0, dtype={'ID': str, 'video_code': str})

    # Open convergence info
    eye_convergence_directory = os.path.join(experiment.subdirs['analysis'], 'eye_convergence')
    convergence_scores_path = os.path.join(eye_convergence_directory, 'convergence_scores.csv')
    convergence_scores = pd.read_csv(convergence_scores_path, dtype={'ID': str})

    # Calculate eye convergence over 20 ms window at beginning and end of bout
    frame_rate = 500.
    window = int(0.02 * frame_rate)

    # Import bout data
    bouts = BoutData.from_directory(bouts_df, experiment.subdirs['kinematics'],
                                    check_tail_lengths=False, tail_columns_only=False)

    # Calculate eye convergence at beginning and end of each bout
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

    # Assign bouts as spontaneous, early, mid, or late prey capture (behaviour phase)
    is_converged = convergence_states[:, 2:].astype('bool')
    is_start = (~is_converged[:, 0]) & is_converged[:, 1]
    is_mid = is_converged[:, 0] & is_converged[:, 1]
    is_end = is_converged[:, 0] & (~is_converged[:, 1])

    behaviour_phase = np.zeros((len(bouts_df),), dtype='i4')
    behaviour_phase[is_start] = 1
    behaviour_phase[is_mid] = 2
    behaviour_phase[is_end] = 3

    bouts_df['phase'] = behaviour_phase
    bouts_df.to_csv(bouts_path, index=True)

    np.save(os.path.join(eye_convergence_directory, 'convergence_states.npy'), convergence_states)
