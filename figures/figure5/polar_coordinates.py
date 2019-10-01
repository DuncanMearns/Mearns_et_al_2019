from plotting import *
from plotting.colors import strike_colors
from datasets.main_dataset import experiment
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


if __name__ == "__main__":

    strike_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'strike_sequences')

    complete_strikes = pd.read_csv(os.path.join(strike_sequence_directory, 'complete_strikes.csv'),
                                   index_col=0, dtype={'ID': str, 'video_code': str})
    strike_labels = complete_strikes['strike_cluster'].values

    all_points = np.load(os.path.join(strike_sequence_directory, 'all_points.npy'))
    forward_points = all_points[(all_points[:, 2] > 10) & (all_points[:, 2] < 50)]
    forward_points = forward_points[(forward_points[:, 3] > -10) & (forward_points[:, 3] < 10)]
    forward_points[:, (2, 3)] -= (10, 0)

    # Test statistic
    attack_labels = np.where(strike_labels == 0)[0]
    sstrike_labels = np.where(strike_labels == 1)[0]
    attack_points = forward_points[np.isin(forward_points[:, 0], attack_labels)]
    sstrike_points = forward_points[np.isin(forward_points[:, 0], sstrike_labels)]

    permutation_test = np.load(os.path.join(strike_sequence_directory, 'permutation_test.npy'))

    for i, param in enumerate(('2D', 'distance', 'angle')):
        edist = permutation_test[i]
        for j, time in enumerate((0, 250, 500)):
            p_value = (edist[1:, j] > edist[0, j]).sum() / float(len(edist) - 1)
            print '{} energy stat; time {}:'.format(param, time), p_value
        print ''

    # fig, axes = plt.subplots(1, 3, figsize=(2, 2), sharey='row', sharex='row')
    # axes = axes[::-1]
    # bins = np.linspace(0, 1.2, 50)
    fig, axes = plt.subplots(3, 1, figsize=(2, 2), sharey='row', sharex='row')
    bins = np.linspace(-60, 60, 50)
    for i, ax in enumerate(axes):

        # # Get points
        # xy_attack = attack_points[attack_points[:, 1] == i][:, (2, 3)]
        # xy_sstrike = sstrike_points[sstrike_points[:, 1] == i][:, (2, 3)]
        # # Calculate distances (0.033 mm / px)
        # d_attack = np.linalg.norm(xy_attack, axis=1) * 0.033
        # d_sstrike = np.linalg.norm(xy_sstrike, axis=1) * 0.033
        # # KDE of distances
        # kde_attack = KernelDensity(bandwidth=0.05).fit(d_attack.reshape((-1, 1)))
        # density_attack = np.exp(kde_attack.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
        # kde_sstrike = KernelDensity(bandwidth=0.05).fit(d_sstrike.reshape((-1, 1)))
        # density_sstrike = np.exp(kde_sstrike.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
        #
        # ax.plot(bins, density_attack, c=strike_colors['attack'], alpha=0.8, lw=1)
        # ax.plot(bins, density_sstrike, c=strike_colors['sstrike'], alpha=0.8, lw=1)
        #
        # open_ax(ax)
        # ax.set_yticks([])
        #
        # ax.set_xlim(0, 1.2)
        # xticks = [0, 0.5, 1.0]
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticks, fontproperties=verysmallfont)
        # ax.set_xticks(np.arange(0, 1.3, 0.1), minor=True)

        # Get points
        xy_attack = attack_points[attack_points[:, 1] == i][:, (2, 3)]
        xy_sstrike = sstrike_points[sstrike_points[:, 1] == i][:, (2, 3)]
        # Calculate distances (0.033 mm / px)
        th_attack = np.degrees(np.arctan2(xy_attack[:, 1], xy_attack[:, 0]))
        th_sstrike = np.degrees(np.arctan2(xy_sstrike[:, 1], xy_sstrike[:, 0]))
        # KDE
        kde_attack = KernelDensity(bandwidth=3.0).fit(th_attack.reshape((-1, 1)))
        density_attack = np.exp(kde_attack.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])
        kde_sstrike = KernelDensity(bandwidth=3.0).fit(th_sstrike.reshape((-1, 1)))
        density_sstrike = np.exp(kde_sstrike.score_samples(bins.reshape((-1, 1)))) * (bins[1] - bins[0])

        ax.plot(bins, density_attack, c=strike_colors['attack'], alpha=0.8, lw=1)
        ax.plot(bins, density_sstrike, c=strike_colors['sstrike'], alpha=0.8, lw=1)

        open_ax(ax)

        ax.set_xlim(-60, 60)
        xticks = np.arange(-60, 70, 30)
        ax.set_xticks(xticks)
        ax.set_xticks(np.arange(-60, 70, 10), minor=True)
        ax.set_xticklabels([])

        ax.set_ylim(0, 0.08)
        yticks = list(np.arange(0, 0.12, 0.05))
        yticks[0] = 0
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontproperties=verysmallfont)
        ax.set_yticks(np.arange(0, 0.1, 0.01), minor=True)

    axes[-1].set_xticklabels(xticks, fontproperties=verysmallfont)



    #     ax.set_xlim(0, 1.2)
    #     xticks = [0, 0.5, 1.0]
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(xticks, fontproperties=verysmallfont)
    #     ax.set_xticks(np.arange(0, 1.3, 0.1), minor=True)
    #
    # axes[-1].set_ylim(0, 0.08)
    # yticks = list(np.arange(0, 0.1, 0.02))
    # yticks[0] = 0
    # axes[-1].set_yticks(yticks)
    # axes[-1].set_yticklabels(yticks, fontproperties=verysmallfont)
    # axes[-1].set_yticks(np.arange(0, 0.08, 0.005), minor=True)

    # plt.show()
    # save_fig(fig, 'figure5', 'angle_distributions')
