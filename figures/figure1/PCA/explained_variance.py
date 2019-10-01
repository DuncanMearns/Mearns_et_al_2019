from plotting import *
from datasets.main_dataset import experiment
import numpy as np

n_components = 8
explained_variance = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'explained_variance.npy'))
explained_variance = explained_variance[:n_components]

if __name__ == "__main__":

    fig_height = 1.5
    fig_width = 1.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                           gridspec_kw=dict(left=0.4, bottom=0.25, right=0.95, top=0.95))

    xpos = np.arange(0.5, n_components + 0.5)
    cumulative_explained_variance = np.cumsum(explained_variance)
    n_pcs = 3

    ax.plot([0, n_pcs - 0.5, n_pcs - 0.5],
            [cumulative_explained_variance[n_pcs - 1], cumulative_explained_variance[n_pcs - 1], 0],
            c='k', ls=':', zorder=0)
    ax.bar(xpos, explained_variance, width=1, color='lightsteelblue', edgecolor='k', zorder=1)
    ax.plot(xpos, cumulative_explained_variance, c='k', zorder=2)
    ax.scatter(xpos, cumulative_explained_variance, c='k', zorder=3, s=10)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontproperties=smallfont)
    ax.set_ylabel('Explained variance', fontproperties=smallfont)

    ax.set_xlim(0, n_components)
    ax.set_xticks(np.arange(n_components) + 0.5)
    ax.set_xticklabels(np.arange(1, n_components + 1), fontproperties=smallfont)
    ax.set_xlabel('Principal\ncomponents (PCs)', fontproperties=smallfont)

    ax.tick_params(axis='x', length=0)
    open_ax()

    print cumulative_explained_variance
    # plt.show()
    save_fig(fig, 'figure1', 'explained_variance')
