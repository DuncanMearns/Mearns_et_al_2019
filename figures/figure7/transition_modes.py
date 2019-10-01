from plotting import *
from plotting.plots import transition_mode_plot
from datasets.lensectomy import experiment
import numpy as np
import os


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')

    for condition in ('control', 'unilateral', 'bilateral'):

        Us, Ss, Vs = np.load(os.path.join(transition_directory, condition, 'USVs.npy'))
        Ua, Sa, Va = np.load(os.path.join(transition_directory, condition, 'USVa.npy'))

        if condition == 'control':
            Va[:, :2] *= (-1, 1)

        fig = transition_mode_plot(Vs, Va)
        save_fig(fig, 'figure7', '{}_transition_modes'.format(condition))
        plt.close(fig)

    # plt.show()
