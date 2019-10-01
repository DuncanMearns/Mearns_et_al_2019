from plotting import *
from plotting.colors import lensectomy_colors, strike_colors
# from plotting.plots.violinplot import violinplot
from datasets.lensectomy import experiment
from matplotlib import gridspec
import numpy as np
import os
import pandas as pd
from scipy.stats import mannwhitneyu


def strike_isomap_plot(ax=None):

    if ax is None:
        ax = plt.gca()

    ax.set_xlim(-500, 250)
    ax.set_ylim(-400, 350)
    open_ax(ax)

    ax.spines['bottom'].set_bounds(-400, 200)
    xticks = np.arange(-400, 400, 200)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontproperties=verysmallfont)
    ax.set_xticks(np.arange(-400, 250, 50), minor=True)

    ax.spines['bottom'].set_color('0.5')
    ax.tick_params(axis='x', which='both', color='0.5', labelcolor='0.5')

    ax.spines['left'].set_bounds(-350, 350)
    yticks = np.arange(-350, 400, 350)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    ax.set_yticks(np.arange(-350, 400, 50), minor=True)

    ax.spines['left'].set_color('0.5')
    ax.tick_params(axis='y', which='both', color='0.5', labelcolor='0.5')


if __name__ == "__main__":

    capture_strike_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes')
    strike_isomap = np.load(os.path.join(capture_strike_directory, 'isomapped_strikes.npy'))
    capture_strikes = pd.read_csv(os.path.join(capture_strike_directory, 'capture_strikes.csv'),
                                  index_col=0, dtype={'ID': str, 'video_code': str})

    strike_labels = np.array(['sstrike', 'attack'])[capture_strikes['cluster_label'].values]
    c = np.array([strike_colors[label] for label in strike_labels])

    control_IDs = experiment.data[experiment.data['condition'] == 'control']['ID']
    unilateral_IDs = experiment.data[experiment.data['condition'].isin(['right', 'left'])]['ID']

    control_strikes = capture_strikes['ID'].isin(control_IDs)
    unilateral_strikes = capture_strikes['ID'].isin(unilateral_IDs)

    # All strikes mapped
    # ------------------

    fig = plt.figure(figsize=(2, 1.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.5)

    ax1 = fig.add_subplot(gs[:2, :2])
    ax1.scatter(*strike_isomap.T, s=2, lw=0, c=c)
    strike_isomap_plot(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    # ax2.scatter(*strike_isomap[~control_strikes].T, s=0.5, lw=0, c='0.75')
    ax2.scatter(*strike_isomap[control_strikes].T, s=0.5, lw=0, c=c[control_strikes])

    ax3 = fig.add_subplot(gs[1, 2])
    # ax3.scatter(*strike_isomap[~unilateral_strikes].T, s=0.5, lw=0, c='0.75')
    ax3.scatter(*strike_isomap[unilateral_strikes].T, s=0.5, lw=0, c=c[unilateral_strikes])

    for ax in (ax2, ax3):
        strike_isomap_plot(ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    save_fig(fig, 'figure7', 'isomapped_strikes')
    # plt.show()
    plt.close(fig)

    # Proportion s-strikes by condition
    # ---------------------------------

    strikes_by_ID = capture_strikes.groupby('ID')['cluster_label']

    control_sstrikes = []
    for ID in control_IDs:
        fish_strikes = strikes_by_ID.get_group(ID)
        n_total = len(fish_strikes)
        n_sstrike = (fish_strikes == 0).sum()
        proportion = float(n_sstrike) / n_total
        control_sstrikes.append(proportion)

    unilateral_sstrikes = []
    for ID in unilateral_IDs:
        try:
            fish_strikes = strikes_by_ID.get_group(ID)
            n_total = len(fish_strikes)
            n_sstrike = (fish_strikes == 0).sum()
            proportion = float(n_sstrike) / n_total
            unilateral_sstrikes.append(proportion)
        except KeyError:
            pass

    fig, ax = plt.subplots(figsize=(1, 1.5))

    vp = ax.violinplot([control_sstrikes, unilateral_sstrikes], showextrema=True, showmedians=True)
    colors = [lensectomy_colors['control'], lensectomy_colors['unilateral']]
    for i, patch in enumerate(vp['bodies']):
        patch.set(facecolor=colors[i], alpha=1, linewidths=0.5, edgecolors='k')
    for line in ('cbars', 'cmaxes', 'cmins', 'cmedians'):
        lc = vp[line]
        lc.set(color='k')

    open_ax()

    ax.set_xticks([])

    print mannwhitneyu(control_sstrikes, unilateral_sstrikes)

    ax.set_ylim(0, 0.8)
    yticks = list(np.arange(0, 1, 0.2))
    yticks[0] = 0
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    ax.set_yticks(np.arange(0, 0.8, 0.05), minor=True)

    save_fig(fig, 'figure7', 'proportion_s-strikes')
    # plt.show()
