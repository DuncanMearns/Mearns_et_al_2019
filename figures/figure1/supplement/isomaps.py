from plotting import *
from datasets.main_dataset import experiment
import os
import numpy as np
from sklearn.neighbors import KernelDensity

from cycler import cycler
colors = plt.cm.Dark2(np.linspace(0,1,4))
plt.rcParams["axes.prop_cycle"] = cycler('color', colors)


if __name__ == "__main__":

    behaviour_space_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space')

    # Load data for isomap with 6 principal components
    more_pc_directory = os.path.join(behaviour_space_directory, 'six_principal_components')
    isomap6 = np.load(os.path.join(more_pc_directory, 'embedding.npy'))[:, :2]
    kernel_pca_eigenvalues = np.load(os.path.join(more_pc_directory, 'eigenvalues.npy'))
    reconstruction_errors = np.load(os.path.join(more_pc_directory, 'reconstruction_errors.npy'))

    # fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))
    # axes[0].plot(kernel_pca_eigenvalues)
    # axes[1].plot(reconstruction_errors)
    # plt.show()

    # Add isomaps with different affinity propagation microclusters
    embedding_directory = os.path.join(behaviour_space_directory, 'embeddings')

    # Plots
    fig, axes = plt.subplots(1, 6, figsize=(7, 1), gridspec_kw=dict(wspace=0.2, left=0.05, right=0.95,
                                                                    bottom=0.05, top=0.9))
    kde = KernelDensity(bandwidth=20).fit(isomap6)
    density = np.exp(kde.score_samples(isomap6))
    axes[0].scatter(*isomap6.T, s=1, lw=0, c=density, cmap='magma', vmin=0)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    open_ax(axes[0])

    preferences = (400, 1000, 2000, 4000)
    i = 0
    for pref, ax in zip(preferences, axes[1:]):
        isomap = np.load(os.path.join(embedding_directory, 'isomap_{}.npy'.format(pref)))[:, :2]
        isomap *= (-1, 1)
        print isomap.shape
        labels = np.load(os.path.join(embedding_directory, 'cluster_labels_{}.npy'.format(pref)))
        sizes = np.array([(labels == l).sum() for l in np.arange(len(isomap))])
        kde = KernelDensity(bandwidth=20).fit(isomap)
        density = np.exp(kde.score_samples(isomap))
        ax.set_title('Affinity={}'.format(pref), fontdict=dict(color=colors[i]), fontproperties=verysmallfont)
        ax.scatter(*isomap.T, lw=0, s=0.01 * sizes, c=density, cmap='magma', vmin=0, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        open_ax(ax)
        i+=1
    axes[1].scatter((-np.ones(4) * 1000) + np.linspace(0, 300, 4), np.ones(4) * 1500, c='k',
                    s=0.01 * np.array([3, 10, 100, 500]), lw=0)

    # Eigenvalues
    eigenvalues = np.load(os.path.join(embedding_directory, 'eigenvalues.npy'))[:4]
    eigenvalues /= eigenvalues[:, [0]]

    ax = axes[-1]
    ax.plot(np.arange(1, 11), eigenvalues.T, alpha=0.5)
    open_ax(ax)

    ax.set_ylim(0, 1)
    yticks = list(np.arange(0, 1.1, 0.2))
    yticks[0] = 0
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)

    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(np.arange(1, 11, 2))
    ax.set_xticklabels(np.arange(1, 11, 2), fontproperties=verysmallfont)
    ax.set_xticks(np.arange(1, 11), minor=True)

    ax.set_title('Eigenvalues', fontproperties=verysmallfont)

    # plt.show()
    # save_fig(fig, 'figureS1', 'compare_isomaps')
    # plt.close(fig)
