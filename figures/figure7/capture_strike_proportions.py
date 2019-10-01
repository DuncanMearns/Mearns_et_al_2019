from plotting import *
from plotting.colors import lensectomy_colors, strike_colors
# from plotting.plots.violinplot import violinplot
from datasets.lensectomy import experiment
from matplotlib import gridspec
import numpy as np
import os
import pandas as pd
from scipy.stats import mannwhitneyu


if __name__ == "__main__":

    capture_strike_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes')
    strike_isomap = np.load(os.path.join(capture_strike_directory, 'isomapped_strikes.npy'))
    capture_strikes = pd.read_csv(os.path.join(capture_strike_directory, 'capture_strikes.csv'),
                                  index_col=0, dtype={'ID': str, 'video_code': str})

    strike_labels = np.array(['sstrike', 'attack'])[capture_strikes['cluster_label'].values]
    c = np.array([strike_colors[label] for label in strike_labels])

    control_IDs = experiment.data[experiment.data['condition'] == 'control']['ID']
    unilateral_IDs = experiment.data[experiment.data['condition'].isin(['right', 'left'])]['ID']

    # Number strikes by condition
    # ---------------------------

    strikes_by_ID = capture_strikes.groupby('ID')['cluster_label']

    control_strikes = []
    for ID in control_IDs:
        fish_strikes = strikes_by_ID.get_group(ID)
        n_sstrike = (fish_strikes == 0).sum()
        n_attack = (fish_strikes == 1).sum()
        control_strikes.append((n_attack, n_sstrike))
    control_strikes = np.array(control_strikes)

    unilateral_strikes = []
    for ID in unilateral_IDs:
        try:
            fish_strikes = strikes_by_ID.get_group(ID)
            n_sstrike = (fish_strikes == 0).sum()
            n_attack = (fish_strikes == 1).sum()
            unilateral_strikes.append((n_attack, n_sstrike))
        except KeyError:
            unilateral_strikes.append((0, 0))
    unilateral_strikes = np.array(unilateral_strikes)

    print 'Attack swims'
    print mannwhitneyu(control_strikes[:, 0], unilateral_strikes[:, 0])
    print 'S-strikes'
    print mannwhitneyu(control_strikes[:, 1], unilateral_strikes[:, 1])
    print '\nControl:',
    print np.median(control_strikes, axis=0)
    print 'Unilateral:',
    print np.median(unilateral_strikes, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(2.0, 1.5))

    vp0 = axes[0].violinplot([control_strikes[:, 0], unilateral_strikes[:, 0]], showextrema=True, showmedians=True)
    vp1 = axes[1].violinplot([control_strikes[:, 1], unilateral_strikes[:, 1]], showextrema=True, showmedians=True)

    colors = [lensectomy_colors['control'], lensectomy_colors['unilateral']]
    for vp in (vp0, vp1):
        for i, patch in enumerate(vp['bodies']):
            patch.set(facecolor=colors[i], alpha=1, linewidths=0.5, edgecolors='k')
        for line in ('cbars', 'cmaxes', 'cmins', 'cmedians'):
            lc = vp[line]
            lc.set(color='k')

    for ax in axes:
        open_ax(ax)
        ax.set_xticks([])
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(0, 80, 20))
        ax.set_yticklabels(np.arange(0, 80, 20), fontproperties=verysmallfont)
        ax.set_xlim(0.5, 2.5)

    axes[1].set_yticklabels([])

    # save_fig(fig, 'figure7', 'strikes_counts_by_fish')
    # plt.show()
