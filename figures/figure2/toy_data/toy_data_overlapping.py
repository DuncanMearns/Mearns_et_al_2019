from plotting import *
from datasets.svd_toy_data.overlapping_states import *
from matplotlib import gridspec
from scipy.cluster import hierarchy as sch
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix

import sys
if __name__ == "__main__":

    # Ground-truth clusters
    fig1, ax = plt.subplots(figsize=(1.5, 1.5))
    xmin, xmax = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
    ymin, ymax = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1
    for c, state in zip(('c', 'm', 'y'), states):
        kde = KernelDensity(bandwidth=0.07).fit(points[np.isin(cluster_labels, state)])
        X, Y = np.meshgrid(np.arange(xmin, xmax + 0.05, 0.05), np.arange(ymin, ymax + 0.05, 0.05))
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        density = np.exp(kde.score_samples(xy))
        ax.contourf(density.reshape(X.shape), extent=(xmin, xmax, ymin, ymax), levels=(1, density.max() + 1),
                    colors=c, alpha=0.5)
        ax.contour(density.reshape(X.shape), extent=(xmin, xmax, ymin, ymax), levels=(1, density.max() + 1),
                   colors=c, linewidths=0.5)
    ax.scatter(*points.T, c='k', s=1, lw=0)
    ax.set_xticks([])
    ax.set_yticks([])
    open_ax(ax)
    save_fig(fig1, 'figureS2', 'overlapping_ground_truth')

    for i, dataset in enumerate([transition_structure_3, transition_structure_4]):

        # Generate transition matrix
        data = ToyData(**dataset)
        T = data.markov_process(n_transitions=10000)

        # Add noise
        noise = np.random.rand(*T.shape)
        noise /= noise.sum()
        T += (noise * T.sum() * 0.25)

        # Analyse transition matrix with SVD
        analysis = TransitionStructureAnalysis(points, T)
        analysis.redistribute_transitions(0.02)
        analysis.svd()
        analysis.transition_space(dataset['ns'], dataset['na'])
        analysis.dendrogram()
        predicted_labels = sch.fcluster(analysis.Y, 6, criterion='maxclust') - 1
        Z = sch.dendrogram(analysis.Y, no_plot=True)
        idxs = Z['leaves']
        M = analysis.T_[:, idxs][idxs]

        # Plot singular values
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1))
        # Symmetric
        ax1.plot(np.arange(10), analysis.Ss[:10], c='k', lw=1, zorder=0)
        ax1.scatter(np.arange(10), analysis.Ss[:10], c='w', edgecolor='k', s=5, zorder=1)
        open_ax(ax1)
        ymin, ymax, ystep = 0, 40, 10
        ax1.set_ylim(ymin, ymax)
        yticks = np.arange(ymin, ymax + ystep, ystep)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticks, fontproperties=verysmallfont)
        # Antisymmetric
        ax2.plot(np.arange(1, 11), analysis.Sa[:20:2], c='k', lw=1, zorder=0)
        ax2.scatter(np.arange(1, 11), analysis.Sa[:20:2], c='w', edgecolor='k', s=5, zorder=1)
        open_ax(ax2)
        ymin, ymax, ystep = 0, 6, 3
        ax2.set_ylim(ymin, ymax)
        yticks = np.arange(ymin, ymax + ystep, ystep)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(yticks, fontproperties=verysmallfont)
        # X-axis
        for ax in (ax1, ax2):
            ax.set_xticks(np.arange(1, 10, 2))
            ax.set_xticklabels(np.arange(1, 10, 2), fontproperties=verysmallfont)
        save_fig(fig2, 'figureS2', 'overlapping_singular_values_{}'.format(i))

        # Plot transition space
        fig3, ax = plt.subplots(figsize=(1, 1))
        ax.scatter(*analysis.q[:, :2].T, c=predicted_labels, cmap='gist_rainbow', s=3, lw=0)
        open_ax(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        save_fig(fig3, 'figureS2', 'overlapping_transition_space_{}'.format(i))

        # Plot transition matrix
        fig4, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(M, cmap='plasma', vmin=0, vmax=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        save_fig(fig4, 'figureS2', 'overlapping_transition_matrix_{}'.format(i), ext='png')

        # Plot identified clusters
        fig5, ax = plt.subplots(figsize=(1, 1))
        ax.scatter(*points.T, c=predicted_labels, s=1, lw=0, cmap='gist_rainbow')
        ax.set_xticks([])
        ax.set_yticks([])
        open_ax(ax)
        save_fig(fig5, 'figureS2', 'overlapping_clustered_{}'.format(i))

        conf = confusion_matrix(true_labels, predicted_labels)
        fig6, ax = plt.subplots(figsize=(1, 1))
        ax.matshow(conf, cmap='binary', vmin=0)
        ax.set_xticks([])
        ax.set_yticks([])
        save_fig(fig6, 'figureS2', 'overlapping_confusion_matrix_{}'.format(i))

    # plt.show()
