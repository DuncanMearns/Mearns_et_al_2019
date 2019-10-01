from plotting import *
from datasets.main_dataset import experiment
import numpy as np
from sklearn.neighbors import KernelDensity


if __name__ == "__main__":

    umap_directory = os.path.join(experiment.subdirs['analysis'], 'umap_embedding')

    fig, axes = plt.subplots(1, 4, figsize=(7, 1.5), gridspec_kw=dict(wspace=0.1, left=0.05, right=0.95,
                                                                      bottom=0.05, top=0.9))
    for n, ax in zip((5, 10, 20, 50), axes):
        umap = np.load(os.path.join(umap_directory, 'umap_embedding_{}.npy'.format(n)))
        kde = KernelDensity(bandwidth=0.5).fit(umap)
        density = np.exp(kde.score_samples(umap))
        ax.set_title('{} nearest neighbors'.format(n), fontproperties=verysmallfont)
        ax.scatter(*umap.T, lw=0, s=3, c=density, cmap='magma', vmin=0)
        ax.set_xticks([])
        ax.set_yticks([])
        open_ax(ax)
    plt.show()
    # save_fig(fig, 'figureS1', 'umap_embeddings')
    # plt.close(fig)
