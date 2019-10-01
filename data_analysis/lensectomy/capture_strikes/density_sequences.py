from datasets.lensectomy import experiment
from stimulus_sequences import hunting_sequence_directory
from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
from behaviour_analysis.miscellaneous import print_subheading
import os
import pandas as pd
import numpy as np


if __name__ == "__main__":

    # Open bouts
    hunting_sequences = pd.read_csv(os.path.join(hunting_sequence_directory, 'hunting_sequences.csv'),
                                    index_col=0, dtype={'ID': str, 'video_code': str})

    right_IDs = experiment.data.groupby('condition')['ID'].get_group('right')
    unilateral_idxs = experiment.data[experiment.data['condition'].isin(['right', 'left'])].index
    experiment.data.loc[unilateral_idxs, 'condition'] = 'unilateral'

    for condition, IDs in experiment.data.groupby('condition')['ID']:

        print_subheading(condition)
        condition_sequences = hunting_sequences[hunting_sequences['ID'].isin(IDs)]

        # Calculate 2d histograms
        histograms = {key: np.zeros((500, 250, 250)) for key in ['attack', 'sstrike']}
        # Calculate pixel averages
        averages = {key: [] for key in ['attack', 'sstrike']}
        counts = {key: [] for key in ['attack', 'sstrike']}

        for ID, fish_bouts in condition_sequences.groupby('ID'):

            print ID

            fish_hunting_sequences = np.load(os.path.join(hunting_sequence_directory, ID + '.npy'))
            assert len(fish_hunting_sequences) == len(fish_bouts)

            if np.isin(ID, right_IDs):  # mirror fish with right lens removed
                fish_hunting_sequences = fish_hunting_sequences[:, :, ::-1]  # TODO: CHECK THAT THIS IS CORRECT!

            # Boolean masks for grabbing frames
            masks = {'attack': (fish_bouts['strike_label'] == 1),
                     'sstrike': (fish_bouts['strike_label'] == 0)}

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
            np.save(os.path.join(hunting_sequence_directory, '{}_{}_histogram.npy'.format(condition, key)),
                    histograms[key])
            # Save averages
            fish_averages = np.array(averages[key])
            fish_counts = np.array(counts[key], dtype='float64')
            if not np.all(fish_counts == 0):
                fish_counts /= np.sum(fish_counts)
            average = np.einsum('i,ijkl->jkl', fish_counts, fish_averages)
            np.save(os.path.join(hunting_sequence_directory, '{}_{}_average.npy'.format(condition, key)), average)
