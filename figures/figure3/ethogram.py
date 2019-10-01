from plotting import *
from plotting.plots.ethogram import plot_ethogram, plot_ethogram_difference
from datasets.main_dataset import experiment
from matplotlib import gridspec
import numpy as np


modelling_directory = os.path.join(experiment.subdirs['analysis'], 'modelling')

T = np.load(os.path.join(modelling_directory, 'T_clusters.npy'))
P = T / T.sum(axis=0)

S = np.load(os.path.join(modelling_directory, 'S_clusters.npy'))
S = S.mean(axis=0)
P_shuffled = S / S.sum(axis=0)

p_decrease, p_increase, sig_decrease, sig_increase = np.load(os.path.join(modelling_directory, 'p-+.npy'))


if __name__ == "__main__":

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 2, 0, 0, 1, 1, 0, 0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    print np.round(P, 2)
    plot_ethogram(ax1, P, False)

    ratio = np.log2(P / P_shuffled)
    print np.round(P / P_shuffled, 2)
    plot_ethogram_difference(ax3, ratio, sig_increase)

    # Key probabilities
    for p, y in zip([0.05, 0.1, 0.2, 0.4, 0.8], [0.8, 0.6, 0.4, 0.2, 0]):
        width = p * 0.2
        head_width = width * 3
        head_length = min(head_width * 1.5, 0.3)
        ax2.arrow(-0.5, y, 1, 0,
                      width=width,
                      length_includes_head=True,
                      head_width=head_width,
                      head_length=head_length,
                      color='0.7',
                      alpha=1,
                      linewidth=0,
                      shape='left')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')

    # Key difference
    for p, y in zip(np.log2([1.125, 1.25, 1.5, 2]), [0.8, 0.6, 0.4, 0.2]):
        width = p * 0.1
        head_width = width * 3
        head_length = min(head_width * 1.5, 0.3)
        ax4.arrow(-0.5, y, 1, 0,
                  width=width,
                  length_includes_head=True,
                  head_width=head_width,
                  head_length=head_length,
                  color='k',
                  alpha=1,
                  linewidth=0,
                  shape='left')
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.axis('off')

    plt.show()
    # save_fig(fig, 'figure3', 'ethogram')
