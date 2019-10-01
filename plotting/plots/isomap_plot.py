from .. import *
from datasets.main_dataset import experiment
import numpy as np
import os


isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))


def isomap_plot(ax=None, dims=(0, 1), scale=1, **kwargs):
    if ax is None:
        ax = plt.gca()

    dict(marker='.',
         s=5 * (scale ** 2),
         linewidths=0)

    ax.scatter(*isomap[:, dims].T, marker='.', s=5*(scale**2), linewidths=0, **kwargs)
    ax.axis('equal')
    open_ax(ax)

    # x axis
    ax.set_xlim(-1500, 1300)
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.spines['bottom'].set_bounds(-1200, 1200)
    ax.set_xticks([-1200, 0, 1200])
    ax.set_xticklabels([])
    ax.set_xticks(np.arange(-1200, 1400, 200), minor=True)
    ax.spines['bottom'].set_color('0.5')
    ax.tick_params(axis='x', which='both', color='0.5', labelcolor='0.5')

    # y axis
    ax.set_ylim(-1200, 1600)
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.spines['left'].set_bounds(-1000, 1400)
    ax.set_yticks([-1000, 0, 1000])
    ax.set_yticks(np.arange(-1000, 1600, 200), minor=True)
    ax.set_yticklabels([])
    ax.spines['left'].set_color('0.5')
    ax.tick_params(axis='y', which='both', color='0.5', labelcolor='0.5')
