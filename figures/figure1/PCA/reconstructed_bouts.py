from plotting import *
from plotting.plots.tail import generate_reconstructed_points, plot_reconstructed_points, plot_trajectory
from plotting.plots.kinematics import plot_tail_kinematics
from plotting.plots.plot_3d import ZAxisLeft

from matplotlib import gridspec
import numpy as np

from datasets.main_dataset.example_data import example_bouts, kinematics, tail_columns
from eigenfish import eigenfish, mean, std


example_bouts = example_bouts.iloc[[0, 2, 8]]
kinematics = kinematics[tail_columns]
eigenfish = eigenfish[:3]


if __name__ == "__main__":

    # Plotting
    """
     ___w1___   w2
    h1       | |  h2
    |________| |__|
         ______
        |      |
        |      h3
        |__w3__|

    """

    total_width = 3.5
    figures = []

    for idx, bout_info in example_bouts.iterrows():

        bout = kinematics.loc[bout_info.start:bout_info.end]
        transformed = np.dot((bout.values - mean) / std, eigenfish.T)
        reconstructed = (np.dot(transformed, eigenfish) * std) + mean

        w1 = 0.7 * len(bout) / 50.

        ps = generate_reconstructed_points(reconstructed, 90)
        ax2_xmin = np.floor(ps[:, 0, :].min())
        ax2_xmax = np.ceil(ps[:, 0, :].max())
        ax2_width = ax2_xmax - ax2_xmin
        h = 1.
        w2 = ax2_width / 50.

        spacing = 0.05

        width = w1 + spacing + w2
        wpad = spacing / width

        fig1 = plt.figure(figsize=(width, 1))
        gs1 = gridspec.GridSpec(1, 2, width_ratios=(w1, w2), wspace=wpad, left=0.05, right=0.95, bottom=0.05, top=0.95)
        ax1 = fig1.add_subplot(gs1[0])
        ax2 = fig1.add_subplot(gs1[1])

        plot_tail_kinematics(ax1, np.degrees(bout), k_max=60)

        plot_reconstructed_points(ax2, ps, c_lim=(0.05, 0.2), lw=1)
        ax2.set_xlim(ax2_xmin, ax2_xmax)
        ax2.set_ylim(-50, 0)
        ax2.axis('off')

        fig3, ax4 = plt.subplots(figsize=(w1, 1))
        ps = generate_reconstructed_points(bout.loc[:, tail_columns].values, 0)
        tip_angle = np.degrees(np.arcsin(ps[:, 1, -1] / 50.))
        ax4.plot(bout.index, tip_angle, c='k', lw=2)
        ax4.axis('off')


        fig2 = plt.figure(figsize=(1.5, 1.5))
        gs2 = gridspec.GridSpec(1, 1, left=0.05, right=0.9, bottom=0.1, top=0.95)
        ax3 = fig2.add_subplot(gs2[0], projection='3d')
        ax3 = fig2.add_axes(ZAxisLeft(ax3))

        plot_trajectory(ax3, transformed, c_lim=(0.05, 0.2), lw=2, fill=True,
                        x_lim=(-10, 10), y_lim=(-8, 8), z_lim=(-5, 5))
        ax3.view_init(15, 120)

        xticks = [0]
        yticks = [0]
        zticks = [0]
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([])
        ax3.set_yticks(yticks)
        ax3.set_yticklabels([])
        ax3.set_zticks(zticks)
        ax3.set_zticklabels([])

        figures.append((fig1, fig2, fig3))

    # Re-scale figures

    fig_widths = np.sum([fig1.get_figwidth() for (fig1, fig2, fig3) in figures])
    scale = total_width / fig_widths

    for idx, (fig1, fig2, fig3) in enumerate(figures):

        fig1.set_size_inches(fig1.get_figwidth() * scale, fig1.get_figheight() * scale)
        fig2.set_size_inches(fig2.get_figwidth() * scale, fig2.get_figheight() * scale)
        fig3.set_size_inches(fig3.get_figwidth() * scale, fig3.get_figheight() * scale)

        save_fig(fig1, 'figure1', 'example_bout_{}'.format(idx))
        save_fig(fig2, 'figure1', 'example_trajectory_{}'.format(idx))
        save_fig(fig3, 'figure1', 'example_tail_angle_{}'.format(idx))

    plt.show()
