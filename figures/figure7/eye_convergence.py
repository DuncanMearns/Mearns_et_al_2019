from plotting import *
from plotting.colors import lensectomy_colors
from plotting.plots import boxplot
from datasets.lensectomy import experiment
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":

    eye_convergence_directory = os.path.join(experiment.subdirs['analysis'], 'eye_convergence')

    # Plot eye convergence distributions
    # ----------------------------------

    fig, axes = plt.subplots(3, 1, figsize=(1, 1.15), sharex='col', sharey='col', gridspec_kw=dict(bottom=0, top=1,
                                                                                                  hspace=0.5))

    for ax, condition in zip(axes, ('control', 'unilateral', 'bilateral')):
        conv_dist = np.load(os.path.join(eye_convergence_directory, '{}_convergence_distribution.npy'.format(condition)))
        ax.fill_between(conv_dist[0], conv_dist[2], color=lensectomy_colors[condition])

        ax.set_xlim(-45, 45)
        ax.set_xticks([-45, 0, 45])
        ax.set_xticks(np.arange(-45, 45, 15), minor=True)

        ax.set_ylim(0, 0.1)
        yticks = [0, 0.1]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontproperties=verysmallfont)
        ax.set_yticks(np.arange(0, 0.1, 0.02), minor=True)

        open_ax(ax)

    axes[-1].set_xticklabels([-45, 0, 45], fontproperties=verysmallfont)

    # save_fig(fig, 'figure7', 'eye_convergence_distribution')
    # plt.show()
    plt.close(fig)

    # Plot eye convergence scores
    # ---------------------------

    convergence_scores = pd.read_csv(os.path.join(eye_convergence_directory, 'convergence_scores.csv'))
    unilateral_idxs = convergence_scores.loc[convergence_scores['condition'].isin(('right', 'left'))]['condition'].index
    convergence_scores.loc[unilateral_idxs, 'condition'] = 'unilateral'

    scores_by_condition = convergence_scores.groupby('condition')['score']

    fig, ax = plt.subplots(figsize=(1, 1.5))
    groups = ('control', 'unilateral', 'bilateral')
    group_scores = [scores_by_condition.get_group(group).values for group in groups]
    print [np.median(scores) for scores in group_scores]
    print [len(group) for group in group_scores]
    boxplot(group_scores, group_colors=[lensectomy_colors[group] for group in groups])

    ax.set_xticks([])

    ax.set_ylim(0, 0.4)
    yticks = list(np.arange(0, 0.5, 0.1))
    yticks[0] = 0
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    ax.set_yticks(np.arange(0, 0.4, 0.02), minor=True)

    # save_fig(fig, 'figure7', 'eye_convergence_scores')
    # plt.show()
    plt.close(fig)

    # Plot hunt initiation rate
    # -------------------------

    ctrl_rate = np.load(os.path.join(eye_convergence_directory, 'hunt_initiation', 'control_rate.npy'))
    uni_rate = np.load(os.path.join(eye_convergence_directory, 'hunt_initiation', 'unilateral_rate.npy'))
    bi_rate = np.load(os.path.join(eye_convergence_directory, 'hunt_initiation', 'bilateral_rate.npy'))

    fig, ax = plt.subplots(figsize=(1, 1.5))
    group_rates = [ctrl_rate, uni_rate, bi_rate]
    print [np.median(rate) for rate in group_rates]
    boxplot(group_rates, group_colors=[lensectomy_colors[group] for group in groups])

    ax.set_xticks([])

    ax.set_ylim(0, 15)
    yticks = list(np.arange(0, 16, 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    ax.set_yticks(np.arange(0, 15, 1), minor=True)

    # save_fig(fig, 'figure7', 'hunt_initiation_rate')
    # plt.show()
