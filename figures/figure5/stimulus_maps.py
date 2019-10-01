from plotting import *
from plotting.stimulus_maps import smoothed_histogram
from datasets.main_dataset import experiment
import numpy as np
import os


if __name__ == "__main__":

    capture_strike_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes')

    # Import data

    stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')
    random_histogram = np.load(os.path.join(stimulus_map_directory, 'random_histogram.npy'))
    random_average = np.load(os.path.join(stimulus_map_directory, 'random_average.npy'))
    random_histogram = random_histogram[100:200, 100:150].T
    random_average = random_average[100:150, 100:200]
    random_histogram = smoothed_histogram(random_histogram, random_average, sigma=1.0)

    strike_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'strike_sequences')

    attack_histograms = np.load(os.path.join(strike_sequence_directory, 'attack_histogram.npy'))
    attack_averages = np.load(os.path.join(strike_sequence_directory, 'attack_average.npy'))

    sstrike_histograms = np.load(os.path.join(strike_sequence_directory, 'sstrike_histogram.npy'))
    sstrike_averages = np.load(os.path.join(strike_sequence_directory, 'sstrike_average.npy'))

    for t in (0, 0.25):

        # Take the appropriate frame and crop images
        f = int((500 * t) + 50)
        attack_histogram = attack_histograms[-f, 100:200, 100:150].T
        attack_average = attack_averages[-f, 100:150, 100:200]
        sstrike_histogram = sstrike_histograms[-f, 100:200, 100:150].T
        sstrike_average = sstrike_averages[-f, 100:150, 100:200]

        # Remove fins from histograms
        attack_histogram[:, :30] = 0
        sstrike_histogram[:, :30] = 0
        # Enhance fins in fish averages
        attack_average *= 2
        sstrike_average *= 2
        attack_average[attack_average > 255] = 255
        sstrike_average[sstrike_average > 255] = 255

        # Mask out fish
        attack_histogram = smoothed_histogram(attack_histogram, attack_average, sigma=1.0)
        sstrike_histogram = smoothed_histogram(sstrike_histogram, sstrike_average, sigma=1.0)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(1.4, 1.4), gridspec_kw=dict(bottom=0, top=1, left=0, right=1, wspace=0))

        attack = (attack_histogram - random_histogram).T[::-1]
        axes[0].imshow(attack_average.T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
        axes[0].contourf(attack, cmap='magma', levels=np.linspace(0.0001, max(0.005, attack_histogram.max()), 100))

        sstrike = (sstrike_histogram - random_histogram).T[::-1]
        axes[1].imshow(sstrike_average.T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
        axes[1].contourf(sstrike, cmap='magma', levels=np.linspace(0.0001, max(0.005, sstrike_histogram.max()), 100))

        for ax in axes:
            ax.axis('off')

        # save_fig(fig, 'figure5', 'stimulus_maps_{}'.format(int(t * 1000)))
        # plt.close(fig)

    # plt.show()
