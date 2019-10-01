from plotting import *
from datasets.main_dataset import experiment
import numpy as np


if __name__ == "__main__":

    data_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space')
    eigenvalues = np.load(os.path.join(data_directory, 'kernel_pca_eigenvalues.npy'))
    reconstruction_errors = np.load(os.path.join(data_directory, 'reconstruction_errors.npy'))

    fig, axes = plt.subplots(2, 1, figsize=(1, 2))

    axes[0].plot(np.arange(1, 9), eigenvalues[:8], c='k', zorder=1)
    axes[0].scatter(np.arange(1, 9), eigenvalues[:8], c='w', s=10, lw=0.5, edgecolor='k', zorder=2)
    axes[0].plot([3, 3, 0], [0, eigenvalues[2], eigenvalues[2]], c='k', lw=1, ls=':', zorder=0)
    axes[0].set_ylim(0, 4e8)
    axes[0].set_xticks([1, 3, 5, 7])

    axes[1].plot(np.arange(1, 9), reconstruction_errors[:8], c='k', zorder=1)
    axes[1].scatter(np.arange(1, 9), reconstruction_errors[:8], c='w', s=10, lw=0.5, edgecolor='k', zorder=2)
    axes[1].plot([3, 3, 0], [0, reconstruction_errors[2], reconstruction_errors[2]], c='k', lw=1, ls=':', zorder=0)
    axes[1].set_ylim(5e4, 16e4)
    axes[1].set_xticks([1, 3, 5, 7])

    for ax in axes:
        open_ax(ax)
        ax.set_xlim(0.5, 8.5)

    # plt.show()
    save_fig(fig, 'figure1', 'embedding_stats')
