from datasets.lensectomy import experiment
from stimulus_sequences import hunting_sequence_directory
from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
from behaviour_analysis.miscellaneous import print_subheading, euclidean_to_polar
import os
import pandas as pd
import numpy as np
from behaviour_analysis.statistics import energy_statistic
from scipy.stats import energy_distance as scipy_energy_distance


if __name__ == "__main__":

    n_permutations = 1000000

    # Open bouts
    hunting_sequences = pd.read_csv(os.path.join(hunting_sequence_directory, 'hunting_sequences.csv'),
                                    index_col=0, dtype={'ID': str, 'video_code': str})

    right_IDs = experiment.data.groupby('condition')['ID'].get_group('right')
    unilateral_idxs = experiment.data[experiment.data['condition'].isin(['right', 'left'])].index
    experiment.data.loc[unilateral_idxs, 'condition'] = 'unilateral'

    IDs_by_condition = experiment.data.groupby('condition')['ID']

    # all_points = []
    # bout_idx = 0
    # for condition_idx, condition in enumerate(('control', 'unilateral')):
    #     print_subheading(condition)
    #     IDs = IDs_by_condition.get_group(condition)
    #     condition_bouts = hunting_sequences[hunting_sequences['ID'].isin(IDs)]
    #     for ID, fish_bouts in condition_bouts.groupby('ID'):
    #         print ID
    #         fish_sequences = np.load(os.path.join(hunting_sequence_directory, ID + '.npy'))
    #         assert len(fish_sequences) == len(fish_bouts)
    #         attacks = (fish_bouts['strike_label'] == 1)
    #         attack_frames = fish_sequences[attacks][:, -50]
    #         if np.isin(ID, right_IDs):  # mirror fish with right lens removed
    #             attack_frames = attack_frames[:, ::-1]
    #         for img in attack_frames:
    #             img_points, img_orientations = find_paramecia(img)
    #             img_points -= np.array([125, 125])
    #             bout_idxs = np.ones(img_orientations.shape) * bout_idx
    #             condition_labels = np.ones(img_orientations.shape) * condition_idx
    #             points = np.vstack([condition_labels, bout_idxs, img_points.T]).T
    #             all_points.append(points)
    #             bout_idx += 1
    #     print ''
    # all_points = np.concatenate(all_points, axis=0)
    # np.save(os.path.join(hunting_sequence_directory, 'attack_points_control_unilateral.npy'), all_points)
    all_points = np.load(os.path.join(hunting_sequence_directory, 'attack_points_control_unilateral.npy'))

    forward_points = all_points[(all_points[:, 2] > 10) & (all_points[:, 2] < 50)]
    forward_points = forward_points[(forward_points[:, 3] > -10) & (forward_points[:, 3] < 10)]
    forward_points[:, (2, 3)] -= (10, 0)

    permutation_test = np.empty((3, n_permutations + 1))  # 2D distribution, distance, theta

    # Test statistic
    points_0 = forward_points[forward_points[:, 0] == 0][:, 2:]
    points_1 = forward_points[forward_points[:, 0] == 1][:, 2:]
    e = energy_statistic(points_0, points_1)
    permutation_test[0, 0] = e
    dt0 = euclidean_to_polar(points_0)
    dt1 = euclidean_to_polar(points_1)
    ed = scipy_energy_distance(dt0[:, 0], dt1[:, 0])
    eth = scipy_energy_distance(dt0[:, 1], dt1[:, 1])
    permutation_test[1:, 0] = (ed, eth)

    bout_idxs = np.unique(forward_points[:, 1])
    n0 = len(np.unique(forward_points[forward_points[:, 0] == 0][:, 1]))

    # Null distribution
    for permutation in range(1, n_permutations + 1):
        if permutation % 10000 == 0:
            print 'Permutation number', permutation

        random_order = np.random.permutation(bout_idxs)
        labels_0 = random_order[:n0]
        labels_1 = random_order[n0:]

        points_0 = forward_points[np.isin(forward_points[:, 1], labels_0)][:, 2:]
        points_1 = forward_points[np.isin(forward_points[:, 1], labels_1)][:, 2:]

        e = energy_statistic(points_0, points_1)
        permutation_test[0, permutation] = e

        dt0 = euclidean_to_polar(points_0)
        dt1 = euclidean_to_polar(points_1)
        ed = scipy_energy_distance(dt0[:, 0], dt1[:, 0])
        eth = scipy_energy_distance(dt0[:, 1], dt1[:, 1])
        permutation_test[1:, permutation] = (ed, eth)

    np.save(os.path.join(hunting_sequence_directory, 'permutation_test.npy'), permutation_test)
