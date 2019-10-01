from plotting import *
from plotting.stimulus_maps import StimulusSequencePlot, smoothed_histogram
from plotting.colors import strike_colors
from datasets.main_dataset import experiment
import numpy as np
import os


if __name__ == "__main__":

    stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')
    strike_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'strike_sequences')

    attack_histogram = np.load(os.path.join(strike_sequence_directory, 'attack_histogram.npy'))
    attack_average = np.load(os.path.join(strike_sequence_directory, 'attack_average.npy'))
    attack_histogram = attack_histogram[:, 100:200, 100:150]
    attack_average = attack_average[:, 100:150, 100:200]

    sstrike_histogram = np.load(os.path.join(strike_sequence_directory, 'sstrike_histogram.npy'))
    sstrike_average = np.load(os.path.join(strike_sequence_directory, 'sstrike_average.npy'))
    sstrike_histogram = sstrike_histogram[:, 100:200, 100:150]
    sstrike_average = sstrike_average[:, 100:150, 100:200]

    random_histogram = np.load(os.path.join(stimulus_map_directory, 'random_histogram.npy'))
    random_average = np.load(os.path.join(stimulus_map_directory, 'random_average.npy'))
    random_histogram = random_histogram[100:200, 100:150].T
    random_average = random_average[100:150, 100:200]
    random_histogram = smoothed_histogram(random_histogram, random_average)

    attack = []
    for cropped, average in zip(attack_histogram, attack_average):
        smoothed = smoothed_histogram(cropped.T, average, threshold=20, sigma=1.0)
        diff = (smoothed - random_histogram).T[::-1]
        attack.append(diff)
    attack = np.array(attack)

    sstrike = []
    for cropped, average in zip(sstrike_histogram, sstrike_average):
        smoothed = smoothed_histogram(cropped.T, average, threshold=20, sigma=1.0)
        diff = (smoothed - random_histogram).T[::-1]
        sstrike.append(diff)
    sstrike = np.array(sstrike)

    plot = StimulusSequencePlot(attack, sstrike, attack_average, sstrike_average, height=1.5,
                                label1=dict(name='Attack swim', c=strike_colors['attack']),
                                label2=dict(name='S-strike', c=strike_colors['sstrike']))
    plot.setup_figure()
    # plot.show_frame(0)
    # plot.play()
    plot.save(os.path.join(output_directory, 'video5'))
