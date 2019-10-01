from datasets.lensectomy import experiment
from paths import stimulus_map_directory
from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col=0, dtype={'ID': str, 'video_code': str})
    n_modules = len(mapped_bouts['module'].unique())

    right_IDs = experiment.data.groupby('condition')['ID'].get_group('right')
    unilateral_idxs = experiment.data[experiment.data['condition'].isin(['left', 'right'])].index
    experiment.data.loc[unilateral_idxs, 'condition'] = 'unilateral'

    for condition, IDs in experiment.data.groupby('condition')['ID']:

        condition_bouts = mapped_bouts[mapped_bouts['ID'].isin(IDs)]

        module_histograms = np.zeros((n_modules, 2, 250, 250))
        module_average_maps = [[] for i in range(n_modules)]
        module_fish_counts = [[] for i in range(n_modules)]

        for ID, fish_bouts in condition_bouts.groupby('ID'):
            print ID
            fish_maps = np.load(os.path.join(stimulus_map_directory, ID + '.npy'))
            assert len(fish_maps) == len(fish_bouts)
            if np.isin(ID, right_IDs):  # mirror fish with right lens removed
                fish_maps = fish_maps[:, :, ::-1]
            for i in range(n_modules):
                module_maps = fish_maps[(fish_bouts['module'] == i)]
                # Add points to the histogram
                for frames in module_maps:
                    for j, frame in enumerate(frames):
                        img_points, img_orientations = find_paramecia(frame)
                        img_counts, x, y = np.histogram2d(*img_points.T, bins=[np.arange(0, 251), np.arange(0, 251)])
                        module_histograms[i, j] += img_counts
                # Calculate fish average
                fish_counts = float(len(module_maps))
                if fish_counts > 0:
                    average_map = np.mean(module_maps.astype('float64'), axis=0)
                else:
                    average_map = np.zeros(module_maps.shape[1:])
                module_average_maps[i].append(average_map)
                module_fish_counts[i].append(fish_counts)

        np.save(os.path.join(stimulus_map_directory, '{}_module_histograms.npy'.format(condition)), module_histograms)

        module_average_maps = np.array(module_average_maps)  # (n_modules, n_fish, 2, x, y)
        module_fish_counts = np.array(module_fish_counts)  # (n_modules, n_fish)
        normed = module_fish_counts / float(module_fish_counts.sum())
        module_fish_counts /= module_fish_counts.sum(axis=1, keepdims=True)
        module_averages = np.einsum('ij,ijklm->iklm', module_fish_counts, module_average_maps)
        np.save(os.path.join(stimulus_map_directory, '{}_module_averages.npy'.format(condition)), module_averages)

        total_average = np.einsum('ij,ijklm->klm', normed, module_average_maps)
        total_average = np.mean(total_average, axis=0)
        np.save(os.path.join(stimulus_map_directory, '{}_total_average.npy'.format(condition)), total_average)
