from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
from compute_stimulus_sequences import strike_sequence_directory
import os
import pandas as pd
import numpy as np
from behaviour_analysis.statistics import energy_statistic
from scipy.stats import energy_distance as scipy_energy_distance
from behaviour_analysis.miscellaneous import euclidean_to_polar


if __name__ == "__main__":

    n_permutations = 100000
    complete_strikes = pd.read_csv(os.path.join(strike_sequence_directory, 'complete_strikes.csv'),
                                   index_col=0, dtype={'ID': str, 'video_code': str})

    time_points = np.array([-50, -175, -300])
    # all_points = []
    # seq_idx = 0
    # for ID, fish_bouts in complete_strikes.groupby('ID'):
    #     print ID
    #     fish_sequences =  np.load(os.path.join(strike_sequence_directory, ID + '.npy'))
    #     assert len(fish_sequences) == len(fish_bouts)
    #     for seq in fish_sequences:
    #         for time_idx, time_point in enumerate(time_points):
    #             img = seq[time_point]
    #             img_points, img_orientations = find_paramecia(img)
    #             img_points -= np.array([125, 125])
    #             sidx = np.ones(img_orientations.shape) * seq_idx
    #             tidx = np.ones(img_orientations.shape) * time_idx
    #             # sequence_index, time_index, x, y, orientation
    #             points = np.hstack([np.array([sidx, tidx]).T, img_points, np.array([img_orientations]).T])
    #             all_points.append(points)
    #         seq_idx += 1
    # all_points = np.concatenate(all_points, axis=0)
    # np.save(os.path.join(strike_sequence_directory, 'all_points.npy'), all_points)
    all_points = np.load(os.path.join(strike_sequence_directory, 'all_points.npy'))

    forward_points = all_points[(all_points[:, 2] > 10) & (all_points[:, 2] < 50)]
    forward_points = forward_points[(forward_points[:, 3] > -10) & (forward_points[:, 3] < 10)]
    forward_points[:, (2, 3)] -= (10, 0)

    strike_labels = complete_strikes['strike_cluster'].values
    permutation_test = np.empty((3, n_permutations + 1, 5))  # 2D distribution, distance, theta

    # Test statistic
    labels_0 = np.where(strike_labels == 0)[0]
    labels_1 = np.where(strike_labels == 1)[0]
    points_0 = forward_points[np.isin(forward_points[:, 0], labels_0)]
    points_1 = forward_points[np.isin(forward_points[:, 0], labels_1)]
    for time_point in range(3):
        xy0 = points_0[points_0[:, 1] == time_point][:, (2, 3)]
        xy1 = points_1[points_1[:, 1] == time_point][:, (2, 3)]
        e = energy_statistic(xy0, xy1)
        permutation_test[0, 0, time_point] = e
        dt0 = euclidean_to_polar(xy0)
        dt1 = euclidean_to_polar(xy1)
        ed = scipy_energy_distance(dt0[:, 0], dt1[:, 0])
        eth = scipy_energy_distance(dt0[:, 1], dt1[:, 1])
        permutation_test[1:, 0, time_point] = (ed, eth)

    # Null distribution
    for permutation in range(1, n_permutations + 1):
        if permutation % 1000 == 0:
            print 'Permutation number', permutation

        random_order = np.random.permutation(strike_labels)

        labels_0 = np.where(random_order == 0)[0]
        labels_1 = np.where(random_order == 1)[0]

        points_0 = forward_points[np.isin(forward_points[:, 0], labels_0)]
        points_1 = forward_points[np.isin(forward_points[:, 0], labels_1)]

        for time_point in range(3):

            xy0 = points_0[points_0[:, 1] == time_point][:, (2, 3)]
            xy1 = points_1[points_1[:, 1] == time_point][:, (2, 3)]
            e = energy_statistic(xy0, xy1)
            permutation_test[0, permutation, time_point] = e

            dt0 = euclidean_to_polar(xy0)
            dt1 = euclidean_to_polar(xy1)
            ed = scipy_energy_distance(dt0[:, 0], dt1[:, 0])
            eth = scipy_energy_distance(dt0[:, 1], dt1[:, 1])
            permutation_test[1:, permutation, time_point] = (ed, eth)

    np.save(os.path.join(strike_sequence_directory, 'permutation_test.npy'), permutation_test)
