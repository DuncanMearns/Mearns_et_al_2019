from plotting import *
from plotting.plots.tail import generate_reconstructed_points, plot_reconstructed_points
from datasets.main_dataset import experiment
import numpy as np
from matplotlib import gridspec

data_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space')
eigenfish = np.load(os.path.join(data_directory, 'eigenfish.npy'))
mean, std = np.load(os.path.join(data_directory, 'tail_statistics.npy'))
n_components = 3

if __name__ == "__main__":

    fig_height = 0.9108 * 2
    fig_width = n_components * fig_height * 0.618 * 2

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, n_components, left=0, right=1, bottom=0.05, top=0.95, wspace=0)
    axes = [fig.add_subplot(gs[i]) for i in range(n_components)]

    for pc, ax in enumerate(axes):
        eigenseries = np.array([eigenfish[pc] * i for i in np.linspace(0, 10, 50)])
        eigenseries = (eigenseries * std) + mean
        tail_points = generate_reconstructed_points(eigenseries, 90)
        plot_reconstructed_points(ax, tail_points, color='binary', lw=2)
        ax.set_xlim(-50, 0)
        ax.axis('equal')
        ax.axis('off')

    # plt.show()
    save_fig(fig, 'figure1', 'eigenfish')
