from datasets.main_dataset import experiment

from plotting import *
from plotting.colors import strike_colors
from plotting.plots.tail import generate_reconstructed_points, plot_reconstructed_points, trajectory_to_bout
from plotting.plots.kinematics import plot_tail_kinematics

from behaviour_analysis.analysis.bouts import BoutData, whiten_data, map_data
from behaviour_analysis.analysis.alignment.dynamic_time_warping import dtw, dtw_path

import numpy as np
import pandas as pd


if __name__ == "__main__":

    # =========
    # Open data
    # =========
    capture_strike_info = pd.read_csv(
        os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'capture_strikes.csv'),
        index_col=0, dtype={'ID': str, 'video_code': str})

    eigenfish = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'eigenfish.npy'))
    mean, std = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'tail_statistics.npy'))

    # =======================================================
    # Load bout data for capture strikes and project onto PCs
    # =======================================================
    capture_strikes = BoutData.from_directory(capture_strike_info, experiment.subdirs['kinematics'],
                                              check_tail_lengths=False, tail_columns_only=True)
    transformed_strikes = capture_strikes.map(eigenfish, whiten=True, mean=mean, std=std)
    transformed_strikes = transformed_strikes.list_bouts(values=True, ndims=3)

    # =========================================================
    # Split strikes by cluster and flip to be in same direction
    # =========================================================
    clustered = [[], []]
    for l, strike in zip(capture_strike_info['strike_cluster'].values, transformed_strikes):
        truncated = strike[12:37]
        truncated = truncated * np.sign(truncated[:10, 1][np.abs(truncated[:10, 1]).argmax()])
        clustered[l].append(truncated)
    cluster_mean_traces = [np.mean(traces, axis=0) for traces in clustered]

    # ==============================================================
    # Align all traces to cluster average and calculate new averages
    # ==============================================================
    cluster_bouts = []
    cluster_averages = []
    for mean_trace, cluster_traces in zip(cluster_mean_traces, clustered):
        aligned_bouts = np.zeros((len(cluster_traces), len(mean_trace), 3))
        bouts = []
        for i, trace in enumerate(cluster_traces):
            x1, y1, d1 = dtw_path(mean_trace, trace, fs=500., bandwidth=0.006)
            x2, y2, d2 = dtw_path(mean_trace, -trace, fs=500., bandwidth=0.006)
            if d1 < d2:
                x_, y_ = x1, y1
                bouts.append(trace)
            else:
                x_, y_ = x2, y2
                bouts.append(-trace)
            for t in np.arange(len(mean_trace)):
                aligned_bouts[i, t] = y_[x_ == t].mean(axis=0)
        average = aligned_bouts.mean(axis=0)
        smoothed_average = np.array([np.convolve(x, np.ones((3,)) / 3., mode='same') for x in average.T]).T
        cluster_bouts.append(bouts)
        cluster_averages.append(smoothed_average)
    cluster_averages = np.array(cluster_averages)

    # Plot aligned trajectories in PC space
    # -------------------------------------
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection='3d')
    for l in range(2):
        for bout in cluster_bouts[l][::3]:
            ax.plot(*bout.T, c=strike_colors[['attack', 'sstrike'][l]], lw=0.5, alpha=0.5, zorder=l)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-10, 10)
    ax.view_init(15, 30)
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_zticks([0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    save_fig(fig, 'figure5', 'postural_dynamics')
    plt.close(fig)

    # =================================================
    # Find the most representative bout in each cluster
    # =================================================
    cluster_bout_idxs = [cluster_info.index for l, cluster_info in capture_strike_info.groupby('strike_cluster')]
    exemplars_kinematics = []

    representative_indices = []
    for cluster, bout_idxs, average in zip(cluster_bouts, cluster_bout_idxs, cluster_averages):
        distances = [dtw(average, bout, fs=500., bandwidth=0.006) for bout in cluster]
        idx_nearest = np.argsort(distances)[:10]
        representative_indices.append(bout_idxs[idx_nearest])
        bout_idx = bout_idxs[idx_nearest[0]]
        representative_kinematics = capture_strikes.get_bout(bout_index=bout_idx).values[12:37]
        exemplars_kinematics.append(representative_kinematics)
    representative_indices = np.concatenate(representative_indices)
    exemplar_info = capture_strike_info.loc[representative_indices]
    exemplar_info.to_csv(os.path.join(output_directory, 'figure5', 'exemplar_strikes.csv'), index=True)

    # Plot tail kinematics and tail series of representative bouts
    # ------------------------------------------------------------
    fig, rows = plt.subplots(2, 2, figsize=(2, 2), dpi=300)
    for i in range(2):
        ax1 = rows[0][i]
        ax2 = rows[1][i]
        k = exemplars_kinematics[i]
        if i == 0:  # flip the attack swim
            kinematics = -k
        else:
            kinematics = k
        trajectory = map_data(whiten_data(kinematics, mean, std), eigenfish)[:, :3]
        pseudo_bout = trajectory_to_bout(trajectory, eigenfish[:3], mean, std)
        ps = generate_reconstructed_points(pseudo_bout, 90)
        plot_reconstructed_points(ax1, ps, lw=1)
        ax1.axis('equal')
        ax1.axis('off')
        plot_tail_kinematics(ax2, kinematics, fs=500., k_max=np.radians(70))
    save_fig(fig, 'figure5', 'representative_bouts')
