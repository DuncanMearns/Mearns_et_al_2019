from plotting import *
from plotting.plots import transition_mode_plot
from datasets.main_dataset import experiment
import numpy as np


if __name__ == "__main__":

    isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))[:, :2]

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions_old')
    USVs = np.load(os.path.join(transition_directory, 'USVs.npy'))
    USVa = np.load(os.path.join(transition_directory, 'USVa.npy'))

    fig = transition_mode_plot(USVs[2], USVa[2])
    # plt.show()
    save_fig(fig, 'figure2', 'transition_modes')
