from plotting import *
from plotting.colors import isomap_colors
from datasets.main_dataset import experiment
import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib import gridspec


if __name__ == "__main__":

    tsne_directory = os.path.join(experiment.subdirs['analysis'], 'tsne_embedding')

    # Plot t-SNEs
    fig = plt.figure(figsize=(7, 3.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.9)
    gs_ = gridspec.GridSpecFromSubplotSpec(3, 3, gs[0], hspace=0.5, wspace=0.5)
    for i, p in enumerate((10, 20, 50)):
        for j, lr in enumerate((10, 100, 1000)):
            tsne = np.load(os.path.join(tsne_directory, 'tsne_embedding_{}_{}.npy'.format(p, lr)))
            kde = KernelDensity(bandwidth=3).fit(tsne)
            density = np.exp(kde.score_samples(tsne))
            ax = fig.add_subplot(gs_[i, j])
            ax.set_title('p={}, lr={}'.format(p, lr), fontproperties=verysmallfont)
            ax.scatter(*tsne.T, lw=0, s=1, c=density, cmap='magma', vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])
            open_ax(ax)
    ax = fig.add_subplot(gs[1])
    ax.set_title('perplexity={}, learning rate={}'.format(p, lr), fontproperties=verysmallfont)
    ax.scatter(*tsne.T, lw=0.5, s=9, c=isomap_colors.colors, edgecolors=isomap_colors.ecolors)
    ax.set_xticks([])
    ax.set_yticks([])
    open_ax(ax)

    # plt.show()
    save_fig(fig, 'figureS1', 'tsne_embeddings')
    plt.close(fig)
