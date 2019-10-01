from plotting import *
from plotting.colors import lensectomy_colors
from datasets.lensectomy import experiment
from behaviour_analysis.miscellaneous import euclidean_to_polar
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


if __name__ == "__main__":

    strike_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'hunting_sequences')

    complete_strikes = pd.read_csv(os.path.join(strike_sequence_directory, 'hunting_sequences.csv'),
                                   index_col=0, dtype={'ID': str, 'video_code': str})

    all_points = np.load(os.path.join(strike_sequence_directory, 'attack_points_control_unilateral.npy'))
    forward_points = all_points[(all_points[:, 2] > 10) & (all_points[:, 2] < 50)]
    forward_points = forward_points[(forward_points[:, 3] > -10) & (forward_points[:, 3] < 10)]
    forward_points[:, (2, 3)] -= (10, 0)

    control_points = forward_points[forward_points[:, 0] == 0][:, 2:]
    unilateral_points = forward_points[forward_points[:, 0] == 1][:, 2:]

    control_d_th = euclidean_to_polar(control_points)
    unilateral_d_th = euclidean_to_polar(unilateral_points)

    # permutation_test = np.load(os.path.join(strike_sequence_directory, 'permutation_test.npy'))
    # for i, param in enumerate(('2D', 'distance', 'angle')):
    #     edist = permutation_test[i]
    #     for j, time in enumerate((0, 100, 200, 300, 400)):
    #         p_value = (edist[1:, j] > edist[0, j]).sum() / float(len(edist) - 1)
    #         print '{} energy stat; time {}:'.format(param, time), p_value
    #     print ''

    fig, axes = plt.subplots(2, 1, figsize=(1, 2))

    # Distances
    control_d = control_d_th[:, 0] * 0.033
    unilateral_d = unilateral_d_th[:, 0] * 0.033
    # KDE of distances
    bins = np.linspace(0, 1.5, 50)
    kde_control = KernelDensity(bandwidth=0.05).fit(control_d.reshape((-1, 1)))
    density_control = np.exp(kde_control.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
    kde_unilateral = KernelDensity(bandwidth=0.05).fit(unilateral_d.reshape((-1, 1)))
    density_unilateral = np.exp(kde_unilateral.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
    # Plot distances
    axes[0].plot(bins, density_control, c=lensectomy_colors['control'], alpha=0.8, lw=1)
    axes[0].plot(bins, density_unilateral, c=lensectomy_colors['unilateral'], alpha=0.8, lw=1)
    # Axes
    open_ax(axes[0])
    # x-axis
    axes[0].set_xlim(0, 1.5)
    xticks = list(np.arange(0, 2.0, 0.5))
    xticks[0] = 0
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticks, fontproperties=verysmallfont)
    axes[0].set_xticks(np.arange(0, 1.5, 0.1), minor=True)
    # y-axis
    axes[0].set_ylim(0, 0.1)
    yticks = list(np.arange(0, 0.12, 0.05))
    yticks[0] = 0
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticks, fontproperties=verysmallfont)
    axes[0].set_yticks(np.arange(0, 0.1, 0.01), minor=True)

    # Angles
    control_th = np.degrees(control_d_th[:, 1])
    unilateral_th = np.degrees(unilateral_d_th[:, 1])
    # KDE of angles
    bins = np.linspace(-60, 60, 50)
    kde_control = KernelDensity(bandwidth=5).fit(control_th.reshape((-1, 1)))
    density_control = np.exp(kde_control.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
    kde_unilateral = KernelDensity(bandwidth=5).fit(unilateral_th.reshape((-1, 1)))
    density_unilateral = np.exp(kde_unilateral.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
    # Plot angles
    axes[1].plot(bins, density_control, c=lensectomy_colors['control'], alpha=0.8, lw=1, zorder=5)
    axes[1].plot(bins, density_unilateral, c=lensectomy_colors['unilateral'], alpha=0.8, lw=1)
    # Axes
    open_ax(axes[1])
    # x-axis
    axes[1].set_xlim(-60, 60)
    xticks = list(np.arange(-60, 70, 30))
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticks, fontproperties=verysmallfont)
    axes[1].set_xticks(np.arange(-60, 60, 10), minor=True)
    # y-axis
    axes[1].set_ylim(0, 0.08)
    yticks = list(np.arange(0, 0.1, 0.04))
    yticks[0] = 0
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(yticks, fontproperties=verysmallfont)
    axes[1].set_yticks(np.arange(0, 0.08, 0.01), minor=True)

    save_fig(fig, 'figure7', 'distance_angular_distributions')
    # plt.show()
