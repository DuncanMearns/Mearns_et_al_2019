from plotting import *
from plotting.stimulus_maps import StimulusSequencePlot, smoothed_histogram
from plotting.colors import lensectomy_colors
from datasets.lensectomy import experiment
import numpy as np
import os


if __name__ == "__main__":

    stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')
    hunting_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'hunting_sequences')

    control_histogram = np.load(os.path.join(hunting_sequence_directory, 'control_attack_histogram.npy'))
    control_average = np.load(os.path.join(hunting_sequence_directory, 'control_attack_average.npy'))
    control_histogram = control_histogram[:, 100:200, 100:150]
    control_average = control_average[:, 100:150, 100:200]

    unilateral_histogram = np.load(os.path.join(hunting_sequence_directory, 'unilateral_attack_histogram.npy'))
    unilateral_average = np.load(os.path.join(hunting_sequence_directory, 'unilateral_attack_average.npy'))
    unilateral_histogram = unilateral_histogram[:, 100:200, 100:150]
    unilateral_average = unilateral_average[:, 100:150, 100:200]

    random_histogram = np.load(os.path.join(stimulus_map_directory, 'random_histogram.npy'))
    random_average = np.load(os.path.join(stimulus_map_directory, 'random_average.npy'))
    random_histogram = random_histogram[100:200, 100:150].T
    random_average = random_average[100:150, 100:200]
    random_histogram = smoothed_histogram(random_histogram, random_average)

    control = []
    for cropped, average in zip(control_histogram, control_average):
        smoothed = smoothed_histogram(cropped.T, average, threshold=20)
        diff = (smoothed - random_histogram).T[::-1]
        control.append(diff)
    control = np.array(control)

    unilateral = []
    for cropped, average in zip(unilateral_histogram, unilateral_average):
        smoothed = smoothed_histogram(cropped.T, average, threshold=20)
        diff = (smoothed - random_histogram).T[::-1]
        unilateral.append(diff)
    unilateral = np.array(unilateral)
    #
    plot = StimulusSequencePlot(control, unilateral, control_average, unilateral_average, height=1.5,
                                label1=dict(name='Sham', c=lensectomy_colors['control']),
                                label2=dict(name='Unilateral', c=lensectomy_colors['unilateral']))
    plot.setup_figure()
    # plot.show_frame(0)
    # plot.play()
    plot.save(os.path.join(output_directory, 'video8'))
