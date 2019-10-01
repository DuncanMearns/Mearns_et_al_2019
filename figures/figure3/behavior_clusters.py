from plotting import *
from plotting.colors import cluster_colors
from plotting.plots import isomap_plot
from plotting.plots.kinematics import plot_tail_kinematics
from plotting.plots.tail import generate_reconstructed_points, plot_reconstructed_points
from behaviour_analysis.analysis.bouts import BoutData
from datasets.main_dataset import experiment
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from matplotlib import gridspec


isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))[:, :3]
exemplars = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'exemplars.csv'),
                        index_col='bout_index', dtype={'ID': str, 'video_code': str})
exemplars = exemplars[exemplars['clean']]


def dtw_1d(s, t, bandwidth=0.01, fs=500.):
    # pad ends with zeros
    n = max([len(s), len(t)])
    t0, t1 = np.zeros((n,)), np.zeros((n,))
    t0[:len(s)] = s
    t1[:len(t)] = t

    # calculate bandwidth
    bw = int(bandwidth * fs)

    # initialise distance matrix
    DTW = np.empty((n, n))
    DTW.fill(np.inf)

    # fill the first row and first column without a cost
    # allows optimal path to be found starting anywhere within the bandwidth
    DTW[0, :bw] = np.array([np.abs(s[0] - t[j]) for j in range(0, bw)])

    # main loop of dtw algorithm
    for i in range(1, n):
        for j in range(max(0, i - bw + 1), min(n, i + bw)):
            DTW[i, j] = np.abs(t0[i] - t1[j]) + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    path = [np.array((n - 1, n - 1))]
    while ~np.all(path[-1] == (0, 0)):
        steps = np.array([(-1, 0), (-1, -1), (0, -1)]) + path[-1]
        if np.any(steps < 0):
            idxs = np.ones((3,), dtype='bool')
            idxs[np.where(steps < 0)[0]] = 0
            steps = steps[idxs]
        path.append(steps[np.argmin(DTW[steps[:, 0], steps[:, 1]])])
    path = np.array(path)[::-1]

    return path[:, 0], t1[path[:, 1]], DTW[-1, -1]


if __name__ == "__main__":

    # Load the exemplar bout data
    exemplars = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'exemplars.csv'), index_col='bout_index',
                            dtype={'ID': str, 'video_code': str})
    exemplars = exemplars[exemplars['clean']]
    labels = exemplars['module'].values

    exemplar_data = BoutData.from_directory(exemplars, experiment.subdirs['kinematics'],
                                            check_tail_lengths=False, tail_columns_only=False)

    eigenfish = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'eigenfish.npy'))
    tail_stats = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'tail_statistics.npy'))

    exemplar_kinematics = BoutData(exemplar_data.data.loc[:, 'k0':'k49'], exemplar_data.metadata)
    exemplars_transformed = exemplar_kinematics.map(eigenfish, True, tail_stats[0], tail_stats[1])

    # Get the confidence that each exemplar belongs to a given cluster
    classifier = NearestNeighbors(10).fit(isomap)
    nearest_neighbors = classifier.kneighbors(isomap, return_distance=False)
    nearest_labels = labels[nearest_neighbors]
    nearest_labels_counts = np.array([np.sum(nearest_labels == l, axis=1) for l in np.unique(labels)]).T
    confidence = nearest_labels_counts[range(len(labels)), labels] / 10.
    colors_alpha = cluster_colors[labels]
    colors_alpha[:, -1] = confidence

    # ============================
    # FIG1: ISOMAP PLOT WITH ALPHA
    # ============================
    fig1, ax = plt.subplots(figsize=(2, 2))
    isomap_plot(ax, scale=2, lw=0.1, c=colors_alpha, edgecolor=cluster_colors[labels])
    # plt.show()
    save_fig(fig1, 'figure3', 'clusters_in_behaviour_space_alpha', 'png')
    # ============================

    # ==========================
    # FIG2: REPRESENTATIVE BOUTS
    # ==========================

    module_names = ['slow1', 'approach', 'turn', 'slow2', 'hat', 'j-turn', 'strike']
    representative_bout_indices = []

    for l, name in enumerate(module_names):

        print name

        bout_idxs = exemplar_data.metadata[labels == l].index
        module_confidence = confidence[labels == l]

        tail_tips = exemplar_data.list_bouts(bout_idxs=bout_idxs)
        tail_tips = [bout['tip'].values for bout in tail_tips]
        flipped = []
        for tail_tip in tail_tips:
            trunc = tail_tip[:25]
            flip = -tail_tip * np.sign(trunc[np.argmax(np.abs(trunc))])
            flipped.append(flip)

        max_length = np.array([len(bout) for bout in flipped]).max()
        padded_tips = np.zeros((len(flipped), max_length)) * np.nan
        for i, bout in enumerate(flipped):
            padded_tips[i, :len(bout)] = bout
        padded_tips = padded_tips * module_confidence.reshape(-1, 1)
        mean = np.nansum(padded_tips, axis=0) / np.sum(module_confidence)

        aligned_tips = np.zeros((len(flipped), len(mean)))
        for i, bout in enumerate(flipped):
            x1, y1, d1 = dtw_1d(mean, bout)
            x2, y2, d2 = dtw_1d(mean, -bout)
            if d1 < d2:
                x_, y_ = x1, y1
            else:
                x_, y_ = x2, y2
            for t in np.arange(len(mean)):
                aligned_tips[i, t] = y_[x_ == t].mean()

        average = np.sum(aligned_tips * module_confidence.reshape(-1, 1), axis=0) / np.sum(module_confidence)
        smoothed_average = np.convolve(average, np.ones((5,)) / 5., mode='same')

        distances = []
        for bout, bout_confidence in zip(flipped, module_confidence):
            x, y, d = dtw_1d(smoothed_average, bout)
            distances.append(d)
        distances_sorted = np.array(distances).argsort()

        nearest_bout = tail_tips[distances_sorted[5]]
        x1, y1, d1 = dtw_1d(smoothed_average, nearest_bout)
        x2, y2, d2 = dtw_1d(smoothed_average, -nearest_bout)
        sign = np.sign(d2 - d1)

        bout_index = bout_idxs[distances_sorted[5]]
        representative_kinematics = exemplar_kinematics.get_bout(bout_index=bout_index).values * sign
        representative_transformed = exemplars_transformed.get_bout(bout_index=bout_index).values[:, :3] * sign
        representative_bout_indices.append(bout_index)

        # ====================
        # Plot data for module
        # ====================

        fig = plt.figure(figsize=(1.5, 1.5))
        gs = gridspec.GridSpec(2, 2, height_ratios=(2, 1), wspace=0.1, hspace=0.1)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Tail tip angles
        for bout, alpha in zip(flipped, confidence):
            ax0.plot(np.degrees(bout), lw=0.1, c=cluster_colors[l], alpha=0.5 * alpha)
        ax0.plot(np.degrees(smoothed_average), lw=1, c='k')
        ax0.set_xlim(0, 100)
        ax0.set_ylim(-240, 240)
        ax0.set_xticks([])
        ax0.set_yticklabels([])
        for spine in ('top', 'right', 'bottom'):
            ax0.spines[spine].set_visible(False)
        ax0.spines['left'].set_color('0.5')
        ax0.tick_params(axis='y', color='0.5')

        # Tail kinematics
        plot_tail_kinematics(ax1, np.degrees(representative_kinematics), ax_lim=(0, 0.18), k_max=120)

        # Pseudo-bout
        k = (np.dot(representative_transformed, eigenfish[:3]) * tail_stats[1]) + tail_stats[0]
        ps = generate_reconstructed_points(k, 90)
        plot_reconstructed_points(ax2, ps[:90], lw=1, c_lim=(0, 0.18))
        ax2.axis('equal')
        ax2.axis('off')

        # plt.show()
        save_fig(fig, 'figure3', name, 'png')
        plt.close(fig)

    representative_bouts = exemplars.loc[representative_bout_indices]
    representative_bouts.to_csv(os.path.join(experiment.subdirs['analysis'],
                                             'clustering', 'representative_bouts.csv'), index=True)
