from plotting import *
from datasets.main_dataset import experiment
import numpy as np


if __name__ == "__main__":

    embedding_directory = os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'embeddings')

    isomap = np.load(os.path.join(embedding_directory, 'isomap_all.npy'))
    tsne = np.load(os.path.join(embedding_directory, 'tsne_embedding.npy'))
    umap = np.load(os.path.join(embedding_directory, 'umap_embedding.npy'))

    fig, axes = plt.subplots(2, 3, figsize=(7, 4), sharex='col', sharey='col',
                             gridspec_kw=dict(wspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.9))

    # Isomap
    axes[0][0].scatter(*isomap.T, s=0.01, lw=0, c='k')
    axes[0][0].set_xlim(-1500, 1500)
    axes[0][0].set_ylim(-1500, 2000)

    counts, xedges, yedges = np.histogram2d(*isomap.T, bins=np.array([np.linspace(-1500, 1500, 100),
                                                                      np.linspace(-1500, 2000, 100)]))
    axes[1][0].imshow(np.log(counts + 1).T, extent=(-1500, 1500, -1500, 2000),
                      cmap='magma', interpolation='bilinear', origin='lower')

    for i, embedding in enumerate([tsne, umap]):
        ax1 = axes[0][i + 1]
        ax1.scatter(*embedding.T, s=0.1, lw=0, c='k')
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        counts, xedges, yedges = np.histogram2d(*embedding.T, bins=np.array([np.linspace(xmin, xmax, 100),
                                                                             np.linspace(ymin, ymax, 100)]))
        ax2 = axes[1][i + 1]
        ax2.imshow(np.log(counts + 1).T, extent=(xmin, xmax, ymin, ymax),
                   cmap='magma', interpolation='bilinear', origin='lower')

    for ax in axes[0]:
        open_ax(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[1]:
        ax.axis('off')

    # plt.show()
    save_fig(fig, 'figureS1', 'embedding_all_points', 'png')
