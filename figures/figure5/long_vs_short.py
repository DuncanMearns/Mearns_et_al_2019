from plotting import *
from plotting.colors import strike_colors
from datasets.main_dataset import experiment
from behaviour_analysis.analysis.bouts import BoutData
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


if __name__ == "__main__":

    # Import data
    capture_strike_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes')
    capture_strikes = pd.read_csv(os.path.join(capture_strike_directory, 'capture_strikes.csv'),
                                  index_col=0, dtype={'ID': str, 'video_code': str})
    strike_frames = pd.read_csv(os.path.join(capture_strike_directory, 'strike_frames.csv'),
                                index_col='bout_index', dtype={'ID': str, 'video_code': str})
    strike_frames = strike_frames[~strike_frames['strike_frame'].isnull()]

    # Dictionary for storing data for each type of strike
    strike_info = dict(attack=dict(), sstrike=dict())
    cluster_labels = dict(attack=0, sstrike=1)

    # Calculate bout lengths
    bout_lengths = (capture_strikes['end'] - capture_strikes['start']).values / 500.
    bins = np.arange(0, 0.3, 0.001).reshape((-1, 1))
    t = bins.squeeze() * 1000
    for key, label in cluster_labels.iteritems():
        strike_lengths = bout_lengths[capture_strikes['strike_cluster'] == label]
        strike_info[key]['lengths'] = strike_lengths
        # Kernel density estimation
        kde = KernelDensity(bandwidth=0.01).fit(strike_lengths.reshape((-1, 1)))
        density = np.exp(kde.score_samples(bins)) * 0.01 * len(strike_lengths) / float(len(bout_lengths))
        strike_info[key]['kde'] = density

    # Capture times
    example_times = (strike_frames['strike_frame'] - strike_frames['start']).values / 500.
    example_lengths = (strike_frames['end'] - strike_frames['start']).values / 500.
    for key, label in cluster_labels.iteritems():
        strike_times = example_times[strike_frames['strike_cluster'] == label]
        kde = KernelDensity(0.01).fit(strike_times.reshape(-1, 1))
        density = np.exp(kde.score_samples(bins)) * 0.01 * len(strike_times) / float(len(example_times))
        strike_info[key]['capture_times'] = strike_times
        strike_info[key]['capture_times_kde'] = density


    # Import kinematic data for example strikes
    example_strike_frames = BoutData.from_directory(strike_frames, experiment.subdirs['kinematics'],
                                                    check_tail_lengths=False, tail_columns_only=False)
    example_capture_frames = example_strike_frames.list_bouts()
    example_capture_frames = [bout.loc[:, 'tip'].values for bout in example_capture_frames]

    # ================
    # COMPOSITE FIGURE
    # ================

    # Make axes
    fig = plt.figure(figsize=(1.854, 3))
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1:3])
    ax3 = fig.add_subplot(gs[3])

    # Plot kde of bout durations
    ax1.fill_between(t, strike_info['attack']['kde'], facecolor=strike_colors['attack'])
    ax1.fill_between(t, strike_info['attack']['kde'], strike_info['attack']['kde'] + strike_info['sstrike']['kde'],
                    facecolor=strike_colors['sstrike'])
    ax1.plot(t, strike_info['attack']['kde'] + strike_info['sstrike']['kde'], c='k', lw=1)

    # Plot example bouts
    colors = []
    for i, angles in enumerate(example_capture_frames):
        x = np.arange(len(angles)) * 2  # n_points * 1000 / 500
        y = np.degrees(angles)
        y *= np.sign(y[np.argmax(np.abs(y[:20]))])
        label = ['attack', 'sstrike'][strike_frames.iloc[i]['strike_cluster']]
        color = strike_colors[label]
        colors.append(color)
        ax2.plot(x, y, c=color, lw=0.1, zorder=1)
    # Plot zero line
    ax2.plot([0, t.max()], [0, 0], lw=1, c='k')
    # Plot bout durations and capture times
    plotting_kwargs = dict(s=10, marker='x', c=colors, lw=0.5, alpha=0.5)
    ax2.scatter(example_times * 1000, np.ones((len(example_times),)) * (-250), **plotting_kwargs)
    ax2.scatter(example_lengths * 1000, np.ones((len(example_lengths),)) * (250), **plotting_kwargs)
    # Plot 50 ms window
    ax2.fill_between(np.array([12, 37]) * 1000 / 500., [-180, -180], [180, 180], facecolor='0.5', alpha=0.5, zorder=0)

    # Plot kde of capture times
    ax3.fill_between(t, strike_info['attack']['capture_times_kde'], facecolor=strike_colors['attack'])
    ax3.fill_between(t, strike_info['attack']['capture_times_kde'],
                     strike_info['attack']['capture_times_kde'] + strike_info['sstrike']['capture_times_kde'],
                     facecolor=strike_colors['sstrike'])
    ax3.plot(t, strike_info['attack']['capture_times_kde'] + strike_info['sstrike']['capture_times_kde'], c='k', lw=1)

    # Fix axes
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, t.max())
    ax1.set_ylim(0, 0.11)
    ax1.set_yticks([0, 0.1])
    ax1.set_yticklabels([0, 0.1])
    ax2.set_ylim(-270, 270)
    ax3.set_ylim(0.27, 0)
    ax3.set_yticks([0, 0.25])
    ax3.set_yticklabels([0, 0.25])
    open_ax(ax1)
    open_ax(ax3)
    for ax in (ax1, ax2):
        ax.set_xticks([])
    ax2.set_yticks([-180, 0, 180])
    ax3.set_xticks([0, 100, 200, 300])

    # plt.show()
    save_fig(fig, 'figure5', 'long_vs_short')
