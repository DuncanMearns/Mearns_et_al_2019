from plotting import *
from plotting.stimulus_maps import smoothed_histogram
from datasets.lensectomy import experiment
import numpy as np
import os


if __name__ == "__main__":

    # Import data

    stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')
    hunting_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'hunting_sequences')

    unilateral_histogram = np.load(os.path.join(hunting_sequence_directory, 'unilateral_attack_histogram.npy'))
    unilateral_average = np.load(os.path.join(hunting_sequence_directory, 'unilateral_attack_average.npy'))

    control_histogram = np.load(os.path.join(hunting_sequence_directory, 'control_attack_histogram.npy'))
    control_average = np.load(os.path.join(hunting_sequence_directory, 'control_attack_average.npy'))

    random_histogram = np.load(os.path.join(stimulus_map_directory, 'random_histogram.npy'))
    random_average = np.load(os.path.join(stimulus_map_directory, 'random_average.npy'))

    # Take the frame of strike onset and crop images

    unilateral_histogram = unilateral_histogram[-50, 100:200, 100:150].T
    unilateral_average = unilateral_average[-50, 100:150, 100:200]
    control_histogram = control_histogram[-50, 100:200, 100:150].T
    control_average = control_average[-50, 100:150, 100:200]
    random_histogram = random_histogram[100:200, 100:150].T
    random_average = random_average[100:150, 100:200]

    # Mask out fish
    unilateral_histogram = smoothed_histogram(unilateral_histogram, unilateral_average)
    control_histogram = smoothed_histogram(control_histogram, control_average)
    random_histogram = smoothed_histogram(random_histogram, random_average)

    fig, axes = plt.subplots(1, 2, figsize=(1.4, 1.4), gridspec_kw=dict(bottom=0, top=1, left=0, right=1, wspace=0))

    control = (control_histogram - random_histogram).T[::-1]
    axes[0].imshow(control_average.T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
    axes[0].contourf(control, cmap='magma', levels=np.linspace(0.0001, 0.005, 100))

    unilateral = (unilateral_histogram - random_histogram).T[::-1]
    axes[1].imshow(unilateral_average.T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
    axes[1].contourf(unilateral, cmap='magma', levels=np.linspace(0.0001, 0.005, 100))

    for ax in axes:
        ax.axis('off')

    # plt.show()
    save_fig(fig, 'figure7', 'prey_density_maps')
