from plotting import *
from plotting.plots import transition_mode_plot
from datasets.spontaneous import experiment
import numpy as np


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')

    USVs = np.load(os.path.join(transition_directory, 'USVs.npy'))
    USVa = np.load(os.path.join(transition_directory, 'USVa.npy'))

    # =====================
    # Plot transition modes
    # =====================
    USVa[2:, :, :2] *= (-1, 1)
    fig1 = transition_mode_plot(USVs[2], USVa[2])

    # ====================
    # Plot singular values
    # ====================
    fig2, axes = plt.subplots(1, 2, figsize=(2, 1))
    # Symmetric
    axes[0].plot(np.arange(10), np.diag(USVs[1])[:10], c='k', zorder=0)
    axes[0].scatter(np.arange(10), np.diag(USVs[1])[:10], s=10, c='w',
                    edgecolor='k', zorder=1)
    axes[0].set_xlim(-0.5, 9.5)
    yticks = np.arange(0, 13, 4).astype('i4')
    axes[0].set_ylim(0, yticks[-1])
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticks, fontproperties=verysmallfont)
    axes[0].set_yticks(np.arange(yticks[0], yticks[-1], 1), minor=True)
    # Anti-symmetric
    axes[1].plot(np.arange(1, 11), np.diag(USVa[1])[:20:2], c='k', zorder=0)
    axes[1].scatter(np.arange(1, 11), np.diag(USVa[1])[:20:2], s=10, c='w',
                    edgecolor='k', zorder=1)
    axes[1].set_xlim(0.5, 10.5)
    yticks = list(np.arange(0, USVa[1, 0, 0] + 0.5, 0.5))
    yticks[0] = 0
    axes[1].set_ylim(0, yticks[-1])
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(yticks, fontproperties=verysmallfont)
    axes[1].set_yticks(np.arange(yticks[0], yticks[-1], 0.1), minor=True)
    # X-axis
    for ax in axes:
        ax.set_xticks(np.arange(1, 10, 2))
        ax.set_xticklabels(np.arange(1, 10, 2), fontproperties=verysmallfont)
        open_ax(ax)

    # plt.show()
    save_fig(fig1, 'figureS3', 'spontaneous_transition_modes')
    save_fig(fig2, 'figureS3', 'spontaneous_singular_values')
