from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.analysis.bouts import BoutData
from behaviour_analysis.miscellaneous import print_heading
from behaviour_analysis.analysis.alignment.distance import calculate_distance_matrix
import numpy as np
import pandas as pd


if __name__ == "__main__":

    frame_rate = 500.
    n_dims = 3

    bout_indices = np.load(paths['bout_indices'])
    bouts_df = pd.read_csv(paths['bouts'], dtype={'ID': str, 'video_code': str})
    bouts_df = bouts_df.loc[bout_indices]
    bouts = BoutData.from_directory(bouts_df, experiment.subdirs['kinematics'],
                                    tail_columns_only=True, check_tail_lengths=False)

    eigenfish = np.load(paths['eigenfish'])
    mean_tail, std_tail = np.load(paths['tail_statistics'])

    transformed = bouts.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)
    transformed_bouts = transformed.list_bouts(values=True, ndims=n_dims)

    print_heading('CALCULATING DISTANCE MATRIX - NORMAL')
    distance_matrix = calculate_distance_matrix(transformed_bouts, fs=frame_rate, flip=False)
    np.save(paths['distance_matrix_normal'], distance_matrix)

    print_heading('CALCULATING DISTANCE MATRIX - FLIPPED')
    distance_matrix = calculate_distance_matrix(transformed_bouts, fs=frame_rate, flip=True)
    np.save(paths['distance_matrix_flipped'], distance_matrix)
