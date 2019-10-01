from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
from behaviour_analysis.miscellaneous import Timer
import numpy as np
import os


if __name__ == "__main__":

    for experiment in (blu, lak):

        stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')

        timer = Timer()
        timer.start()

        # Compute average

        rand = np.load(os.path.join(stimulus_map_directory, 'random_frames.npy'))
        subset_averages = []
        subset_counts = []
        for i in np.arange(0, len(rand), 1000):
            subset = rand[i:i+1000].astype('float64')
            subset_averages.append(subset.mean(axis=0))
            subset_counts.append(float(len(subset)))
        subset_averages = np.array(subset_averages)
        subset_counts = np.array(subset_counts)
        average = np.einsum('i,ijk->jk', subset_counts, subset_averages)
        average /= subset_counts.sum()

        np.save(os.path.join(stimulus_map_directory, 'random_average.npy'), average)

        # Compute histogram

        histogram = np.zeros((250, 250))

        for img in rand:
            img_points, img_orientations = find_paramecia(img)
            img_counts, x, y = np.histogram2d(*img_points.T, bins=[np.arange(0, 251), np.arange(0, 251)])
            histogram += img_counts

        np.save(os.path.join(stimulus_map_directory, 'random_histogram.npy'), histogram)

        print 'Time taken:', timer.convert_time(timer.stop())
