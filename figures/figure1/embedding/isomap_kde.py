from plotting import *
from datasets.main_dataset import experiment
import os
import numpy as np
import pandas as pd
from behaviour_analysis.analysis.embedding import gaussian_kde


if __name__ == "__main__":

    isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))[:, :2]
    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col='bout_index',
                               dtype={'ID': str, 'video_code': str})

    n_states = len(isomap)
    cluster_sizes = np.array([(mapped_bouts['state'] == l).sum() for l in np.arange(n_states)])

    # KDE over whole space
    x1, y1 = np.meshgrid(np.arange(-1500, 1500, 10), np.arange(-1300, 1700, 10))
    xy = np.vstack((x1.ravel(), y1.ravel()))
    pdf_xy = gaussian_kde(isomap[:, [0, 1]].T, bw=0.15, weights=cluster_sizes / float(cluster_sizes.max()))
    density_xy = pdf_xy.evaluate(xy)
    log_xy = np.log(density_xy).reshape(x1.shape)

    # KDE over forward scoots
    x1, y1 = np.meshgrid(np.arange(-500, 0, 5), np.arange(-400, 300, 5))
    xy = np.vstack((x1.ravel(), y1.ravel()))
    pdf_xy = gaussian_kde(isomap[:, [0, 1]].T, bw=0.08, weights=cluster_sizes / float(cluster_sizes.max()))
    density_xy = pdf_xy.evaluate(xy)
    scoot_xy = density_xy.reshape(x1.shape)

    # Plots
    fig1, ax1 = plt.subplots(figsize=(1.5, 1.5))
    ax1.imshow(log_xy, extent=(-1500, 1500, -1300, 1700), cmap='magma', vmin=-20, vmax=-11, origin='lower')
    ax1.plot([-500, -500, 0, 0, -500], [-400, 300, 300, -400, -400], c='w', lw=0.5, ls='--')

    fig2, ax2 = plt.subplots(figsize=(0.7, 1.5))
    ax2.contourf(scoot_xy, cmap='magma', levels=np.linspace(0, 1.2e-5, 20), origin='lower')
    ax2.axis('equal')

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    # plt.show()
    save_fig(fig1, 'figure1', 'isomap_kde')
    save_fig(fig2, 'figure1', 'scoot_kde')
