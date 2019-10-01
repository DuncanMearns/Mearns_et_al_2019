from plotting import *
from plotting.colors import virtual_prey_colors
import numpy as np
import os
from scipy.stats import ks_2samp, wilcoxon


analysis_directory = 'D:\\DATA\\virtual_prey_capture\\analysis'


if __name__ == "__main__":

    # =============
    # Hunt duration
    # =============

    hunt_durations = {}
    fig, ax = plt.subplots(figsize=(1, 1.618))
    bins = np.arange(0, 6.5, 0.5)
    for trial in ('control', 'test'):
        durations = np.load(os.path.join(analysis_directory, '{}_durations.npy'.format(trial)))
        hunt_durations[trial] = durations
        counts, bins = np.histogram(durations, bins=bins)
        cumulative = np.zeros(bins.shape)
        cumulative[1:] = np.cumsum(counts)
        cumulative /= cumulative[-1]
        ax.plot(bins, cumulative, c=virtual_prey_colors[trial])
    open_ax()
    ax.set_xlim(0, 6)
    ax.set_xticks(bins[::4])
    ax.set_xticks(bins, minor=True)
    ax.set_xticklabels(bins[::4].astype('i4'), fontproperties=verysmallfont)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax.set_yticklabels([0, 0.5, 1.0], fontproperties=verysmallfont)
    # plt.show()
    # save_fig(fig, 'figure4', 'virtual_hunt_duration_distributions')

    print 'Hunt durations'
    print ks_2samp(hunt_durations['control'], hunt_durations['test'])
    print ''

    # =====================
    # Hunt duration by fish
    # =====================

    fish_durations = np.load(os.path.join(analysis_directory, 'fish_hunt_durations.npy'))
    print fish_durations.mean(axis=0)

    fig, ax = plt.subplots(figsize=(1, 1.618))
    ax.plot(fish_durations[:, :2].T, c='k', lw=0.5, zorder=0)
    ax.scatter(np.zeros(len(fish_durations),), fish_durations[:, 0],
               zorder=1, c='w',
               edgecolor=virtual_prey_colors['control'], lw=0.5, s=10)
    ax.scatter(np.ones(len(fish_durations), ), fish_durations[:, 1],
               zorder=1, c='w',
               edgecolor=virtual_prey_colors['test'], lw=0.5, s=10)

    open_ax()
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 2.5)
    yticks = np.arange(0, 3, 0.5)
    ax.set_yticks(yticks)
    yticks = list(yticks)
    yticks[0] = 0
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    # plt.show()
    # save_fig(fig, 'figure4', 'virtual_hunt_durations')

    print 'Fish durations'
    print wilcoxon(fish_durations[:, 0], fish_durations[:, 1])
    print ''

    # ================
    # Sequence lengths
    # ================

    sequence_lengths = {}
    fig, ax = plt.subplots(figsize=(1, 1.618))
    bins = np.arange(0, 11)
    for trial in ('control', 'test'):
        lengths = np.load(os.path.join(analysis_directory, '{}_sequences.npy'.format(trial)))
        sequence_lengths[trial] = lengths
        counts, bins = np.histogram(lengths, bins=bins)
        cumulative = np.zeros(bins.shape)
        cumulative[1:] = np.cumsum(counts)
        cumulative /= cumulative[-1]
        ax.plot(bins, cumulative, c=virtual_prey_colors[trial])
    open_ax()
    ax.set_xlim(0, 10)
    ax.set_xticks(bins[::5])
    ax.set_xticks(bins, minor=True)
    ax.set_xticklabels(bins[::5].astype('i4'), fontproperties=verysmallfont)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax.set_yticklabels([0, 0.5, 1.0], fontproperties=verysmallfont)
    # plt.show()
    # save_fig(fig, 'figure4', 'virtual_hunt_sequence_distributions')

    print 'Sequence lengths'
    print ks_2samp(sequence_lengths['control'], sequence_lengths['test'])
    print ''

    # =====================
    # Fish sequence lengths
    # =====================

    fish_lengths = np.load(os.path.join(analysis_directory, 'fish_sequence_lengths.npy'))
    print np.median(fish_lengths, axis=0)

    fig, ax = plt.subplots(figsize=(1, 1.618))
    ax.plot(fish_lengths[:, :2].T, c='k', lw=0.5, zorder=0)
    ax.scatter(np.zeros(len(fish_lengths),), fish_lengths[:, 0],
               zorder=1, c='w',
               edgecolor=virtual_prey_colors['control'], lw=0.5, s=10)
    ax.scatter(np.ones(len(fish_lengths), ), fish_lengths[:, 1],
               zorder=1, c='w',
               edgecolor=virtual_prey_colors['test'], lw=0.5, s=10)

    open_ax()
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 3)
    yticks = np.arange(0, 4)
    ax.set_yticks(yticks)
    yticks = list(yticks)
    yticks[0] = 0
    ax.set_yticklabels(yticks, fontproperties=verysmallfont)
    # plt.show()
    # save_fig(fig, 'figure4', 'virtual_hunt_sequences')

    print 'Fish lengths'
    print wilcoxon(fish_lengths[:, 0], fish_lengths[:, 1])
    print ''
