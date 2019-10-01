from plotting import *
from plotting.stimulus_maps import smoothed_histogram
from datasets.lakritz import experiment
import numpy as np


stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')

# RANDOM MAPS FOR BLU
random_histogram = np.load(os.path.join(stimulus_map_directory, 'random_histogram.npy'))
random_average = np.load(os.path.join(stimulus_map_directory, 'random_average.npy'))[..., 1:251]

ymin, ymax = 50, 200
xmin, xmax = 60, 190
sigma = 2.0

random_histogram = random_histogram[ymin:ymax, xmin:xmax].T
random_average = random_average[xmin:xmax, ymin:ymax]
random = smoothed_histogram(random_histogram, random_average, threshold=30, sigma=sigma)


for condition in ('ctrl', 'mut'):

    averages = np.load(os.path.join(stimulus_map_directory, '{}_module_averages.npy'.format(condition)))[..., 1:251]
    histograms = np.load(os.path.join(stimulus_map_directory, '{}_module_histograms.npy'.format(condition)))

    for i in range(7):

        fig, axes = plt.subplots(1, 3)

        pre_histogram = histograms[i, 0, ymin:ymax, xmin:xmax].T
        post_histogram = histograms[i, 1, ymin:ymax, xmin:xmax].T

        pre_average = averages[i, 0, xmin:xmax, ymin:ymax]
        post_average = averages[i, 1, xmin:xmax, ymin:ymax]

        pre_histogram = smoothed_histogram(pre_histogram, pre_average, threshold=30, sigma=sigma)
        post_histogram = smoothed_histogram(post_histogram, post_average, threshold=30, sigma=sigma)

        pre_density = (pre_histogram - random_histogram).T[::-1]
        post_density = (post_histogram - random_histogram).T[::-1]

        lower = np.percentile(np.array([pre_density, post_density]), 85)
        upper = max([np.array([pre_density, post_density]).max(), 1e-4])

        pre_density[pre_density < lower] = 0
        post_density[post_density < lower] = 0

        axes[0].imshow(pre_average.T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
        axes[0].contourf(pre_density, cmap='magma', levels=np.linspace(2e-5, upper, 50))

        axes[1].imshow(post_average.T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
        axes[1].contourf(post_density, cmap='magma', levels=np.linspace(2e-5, upper, 50))

        difference = post_density - pre_density
        axes[2].imshow(post_average.T[::-1], cmap='binary', vmin=0, vmax=255, interpolation='bilinear')
        axes[2].contourf(difference, cmap='Reds', levels=np.linspace(2e-5, upper, 20))
        axes[2].contourf(difference, cmap='Blues_r', levels=np.linspace(min([-(upper / 2.), difference.min()]), -1e-5, 20))

        for ax in axes:
            ax.axis('off')

        # save_fig(fig, 'figure4', 'lak_{}_stimulus_maps_{}'.format(condition, i))
        # plt.close(fig)

    plt.show()
