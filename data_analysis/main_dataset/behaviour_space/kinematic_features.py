from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.analysis.bouts import BoutData
import pandas as pd
import numpy as np


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(paths['mapped_bouts'], index_col='bout_index', dtype={'ID': str, 'video_code': str})
    n_states = len(mapped_bouts['state'].unique())
    cluster_indices = [np.where(mapped_bouts['state'].values == l)[0] for l in np.arange(n_states)]

    mapped_bouts_data = BoutData.from_directory(mapped_bouts, experiment.subdirs['kinematics'],
                                                tail_columns_only=False, check_tail_lengths=False)
    all_bouts = mapped_bouts_data.list_bouts()

    # Turn angles
    turn_angles = np.array([bout['angular_velocity'].sum() for bout in all_bouts])
    turn_angles = np.degrees(np.abs(turn_angles)) / 500.  # ignore direction, convert to degrees, account for frame rate
    turn_angles = np.array([np.median(turn_angles[idxs]) for idxs in cluster_indices])  # take cluster medians

    # Maximum angular velocity
    max_angular_velocities = np.array([bout['angular_velocity'].max() for bout in all_bouts])
    max_angular_velocities = np.degrees(np.abs(max_angular_velocities))  # ignore direction, convert to degrees
    max_angular_velocities = np.array([np.median(max_angular_velocities[idxs]) for idxs in cluster_indices])

    # Time of maximum angular velocity
    t_max_angular_velocities = np.array([np.argmax(bout['angular_velocity'].values) for bout in all_bouts])
    t_max_angular_velocities = np.abs(t_max_angular_velocities) / 500.  # ignore direction, account for frame rate
    t_max_angular_velocities = np.array([np.median(t_max_angular_velocities[idxs]) for idxs in cluster_indices])

    # Bout mean speed
    mean_speeds = np.array([bout['speed'].mean() for bout in all_bouts])
    mean_speeds = np.array([np.median(mean_speeds[idxs]) for idxs in cluster_indices]) / 500.

    # Save
    kinematic_features = np.array([turn_angles, max_angular_velocities, t_max_angular_velocities, mean_speeds])
    print kinematic_features.shape
    np.save(paths['kinematic_features'], kinematic_features)
