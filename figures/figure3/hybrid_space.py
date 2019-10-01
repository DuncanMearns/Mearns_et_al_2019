from plotting import *
from plotting.colors import cluster_colors
from datasets.main_dataset import experiment
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch


if __name__ == "__main__":

    hybrid_space = np.load(os.path.join(experiment.subdirs['analysis'], 'clustering', 'hybrid_space.npy'))
    silhouette_scores = np.load(os.path.join(experiment.subdirs['analysis'], 'clustering', 'silhouette_scores.npy'))
    exemplars = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'exemplars.csv'), index_col='bout_index',
                            dtype={'ID': str, 'video_code': str})
    exemplars = exemplars[exemplars['clean']]
    labels = exemplars['module'].values

    D1 = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'exemplar_distance_matrix.npy'))
    D2 = np.load(os.path.join(experiment.subdirs['analysis'], 'clustering', 'transition_distances.npy'))

    fig1, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.scatter(*hybrid_space[:, :2].T, c=cluster_colors[labels], lw=0, s=3)
    ax.set_xticks([])
    ax.set_yticks([])
    open_ax()

    fig2, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.plot(np.arange(2, 14), silhouette_scores, c='k')
    xticks = np.arange(2, 15, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontproperties=verysmallfont)
    yticks = np.arange(0.2, 0.26, 0.01)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    open_ax()

    fig3, axes = plt.subplots(2, 1, figsize=(1, 2))
    Y = sch.linkage(hybrid_space, method='ward')
    Z = sch.dendrogram(Y, no_plot=True)
    idxs = Z['leaves']
    axes[0].imshow(D1[:, idxs][idxs], cmap='inferno')
    axes[1].imshow(D2[:, idxs][idxs], cmap='inferno')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.show()
    save_fig(fig1, 'figure3', 'hybrid_space')
    save_fig(fig2, 'figure3', 'silhouette_scores')
    save_fig(fig3, 'figure3', 'distance_matrices')
