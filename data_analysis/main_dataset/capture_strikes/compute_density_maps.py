from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
from compute_stimulus_sequences import strike_sequence_directory
import os
import pandas as pd
import numpy as np


if __name__ == "__main__":

    complete_strikes = pd.read_csv(os.path.join(strike_sequence_directory, 'complete_strikes.csv'),
                                   dtype={'ID': str, 'video_code': str}, index_col='bout_index')

    # Calculate 2d histograms
    histograms = {key: np.zeros((500, 250, 250)) for key in ['attack', 'sstrike']}
    # Calculate pixel averages
    averages = {key: [] for key in ['attack', 'sstrike']}
    counts = {key: [] for key in ['attack', 'sstrike']}

    for ID, fish_bouts in complete_strikes.groupby('ID'):

        print ID

        fish_hunting_sequences = np.load(os.path.join(strike_sequence_directory, ID + '.npy'))
        assert len(fish_hunting_sequences) == len(fish_bouts)

        # Boolean masks for grabbing frames
        masks = {'attack': (fish_bouts['strike_cluster'] == 0),
                 'sstrike': (fish_bouts['strike_cluster'] == 1)}

        for key, mask in masks.iteritems():
            masked = fish_hunting_sequences[mask]
            for seq in masked:
                for i, img in enumerate(seq):
                    img_points, img_orientations = find_paramecia(img)
                    img_counts, x, y = np.histogram2d(*img_points.T, bins=[np.arange(0, 251), np.arange(0, 251)])
                    histograms[key][i] += img_counts

            fish_count = len(masked)
            if len(masked) > 0:
                fish_average = np.mean(masked.astype('float64'), axis=0)
            else:
                fish_average = np.zeros((500, 250, 252))
            averages[key].append(fish_average)
            counts[key].append(fish_count)

    for key in ('attack', 'sstrike'):
        # Save histograms
        np.save(os.path.join(strike_sequence_directory, '{}_histogram.npy'.format(key)),
                histograms[key])
        # Save averages
        fish_averages = np.array(averages[key])
        fish_counts = np.array(counts[key], dtype='float64')
        if not np.all(fish_counts == 0):
            fish_counts /= np.sum(fish_counts)
        average = np.einsum('i,ijkl->jkl', fish_counts, fish_averages)
        np.save(os.path.join(strike_sequence_directory, '{}_average.npy'.format(key)), average)

    # np.save(os.path.join(strike_sequence_directory, 'attack_histogram.npy'), histograms['attack'])
    # np.save(os.path.join(strike_sequence_directory, 'sstrike_histogram.npy'), histograms['sstrike'])
