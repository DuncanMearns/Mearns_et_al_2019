from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.analysis.bouts import BoutData
from behaviour_analysis.analysis.alignment import calculate_distance_matrix
from behaviour_analysis.miscellaneous import print_heading, Timer
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    capture_strike_info = pd.read_csv(paths['capture_strikes'],
                                      index_col='bout_index',
                                      dtype={'ID': str, 'video_code': str})

    # Import bouts
    capture_strikes = BoutData.from_directory(capture_strike_info, experiment.subdirs['kinematics'],
                                              check_tail_lengths=False, tail_columns_only=True)

    # Transform
    mean_tail, std_tail = np.load(os.path.join(experiment.subdirs['analysis'],
                                               'behaviour_space',
                                               'tail_statistics.npy'))
    eigenfish = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'eigenfish.npy'))

    transformed_strikes = capture_strikes.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)
    transformed_strikes = transformed_strikes.list_bouts(values=True, ndims=3)

    # Truncate
    truncated_strikes = [bout[12:37] for bout in transformed_strikes]
    bw = 0.006  # 3 frames

    print_heading('CALCULATING CAPTURE STRIKE DISTANCE MATRIX')
    timer = Timer()
    timer.start()
    S_normal = calculate_distance_matrix(truncated_strikes, bw=bw)
    S_flipped = calculate_distance_matrix(truncated_strikes, bw=bw, flip=True)
    S = np.min([S_normal, S_flipped], axis=0)
    np.save(paths['capture_strike_distance_matrix'], S)
    time_taken = timer.stop()
    print 'Time taken: {}'.format(timer.convert_time(time_taken))
