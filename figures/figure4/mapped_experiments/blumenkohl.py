from plotting import *
from plotting.colors import mutant_colors
from plotting.plots import transition_mode_plot
from datasets.blumenkohl import experiment
import numpy as np


if __name__ == "__main__":

    ylimits = dict(het=dict(ymin0=-10, ymax0=30, ystep0=10,
                            ymin1=-2, ymax1=1, ystep1=1),
                   mut=dict(ymin0=-10, ymax0=20, ystep0=10,
                            ymin1=-8, ymax1=0, ystep1=2))

    for condition in ('het', 'mut'):

        transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')

        USVs = np.load(os.path.join(transition_directory, condition, 'USVs.npy'))
        USVa = np.load(os.path.join(transition_directory, condition, 'USVa.npy'))

        # =====================
        # Plot transition modes
        # =====================
        fig1 = transition_mode_plot(USVs[2], USVa[2])

        # ====================
        # Plot singular values
        # ====================
        fig2, axes = plt.subplots(1, 2, figsize=(2, 1))
        # Symmetric
        axes[0].plot(np.arange(10), np.diag(USVs[1])[:10], c=mutant_colors['blu_s257'][condition], zorder=0)
        axes[0].scatter(np.arange(10), np.diag(USVs[1])[:10], s=10, c='w',
                        edgecolor=mutant_colors['blu_s257'][condition], zorder=1)
        axes[0].set_xlim(-0.5, 9.5)
        yticks = np.arange(0, USVs[1, 0, 0] + 5, 5).astype('i4')
        axes[0].set_ylim(0, yticks[-1])
        axes[0].set_yticks(yticks)
        axes[0].set_yticklabels(yticks, fontproperties=verysmallfont)
        axes[0].set_yticks(np.arange(yticks[0], yticks[-1], 1), minor=True)
        # Anti-symmetric
        axes[1].plot(np.arange(1, 11), np.diag(USVa[1])[:20:2], c=mutant_colors['blu_s257'][condition], zorder=0)
        axes[1].scatter(np.arange(1, 11), np.diag(USVa[1])[:20:2], s=10, c='w',
                        edgecolor=mutant_colors['blu_s257'][condition], zorder=1)
        axes[1].set_xlim(0.5, 10.5)
        yticks = list(np.arange(0, USVa[1, 0, 0] + 1, 1).astype('i4'))
        yticks[0] = 0
        axes[1].set_ylim(0, yticks[-1])
        axes[1].set_yticks(yticks)
        axes[1].set_yticklabels(yticks, fontproperties=verysmallfont)
        axes[1].set_yticks(np.arange(yticks[0], yticks[-1], 0.2), minor=True)
        # X-axis
        for ax in axes:
            ax.set_xticks(np.arange(1, 10, 2))
            ax.set_xticklabels(np.arange(1, 10, 2), fontproperties=verysmallfont)
            open_ax(ax)

        # =====================
        # Plot prediction error
        # =====================

        sym = np.load(os.path.join(transition_directory, condition, 'symmetric_models.npy'))
        asym = np.load(os.path.join(transition_directory, condition, 'antisymmetric_models.npy'))
        asym = np.concatenate([sym[:, [0]], asym], axis=1)  # Add the null model to the antisymmetric models

        fig3, axes = plt.subplots(1, 2, figsize=(2, 1))
        for ax, models in zip(axes, [sym, asym]):
            improvement = 100 * (models[:, [0]] - models[:, 1:]) / models[:, [0]]
            mean_improvement = np.mean(improvement, axis=0)
            std_improvement = np.std(improvement, axis=0)
            ax.scatter(np.arange(1, 6), mean_improvement, s=10, lw=1, c='w',
                       edgecolor=mutant_colors['blu_s257'][condition], zorder=1)
            ax.errorbar(np.arange(1, 6), mean_improvement, yerr=std_improvement,
                        capsize=3, fmt='none', zorder=0, c=mutant_colors['blu_s257'][condition], lw=1)
            open_ax(ax)
            ax.set_xlim(0.5, 5.5)
            ax.set_xticks(np.arange(1, 6))
            ax.set_xticklabels(np.arange(1, 6), fontproperties=verysmallfont)
            print mean_improvement

        for i, ax in enumerate(axes):
            ymin = ylimits[condition]['ymin' + str(i)]
            ymax = ylimits[condition]['ymax' + str(i)]
            ystep = ylimits[condition]['ystep' + str(i)]
            ax.set_ylim(ymin, ymax)
            yticks = np.arange(ymin, ymax + ystep, ystep)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, fontproperties=verysmallfont)
            ax.set_yticks(np.arange(ymin, ymax, ystep / 5.), minor=True)

        # plt.show()
        save_fig(fig1, 'figure4', 'blu_{}_transition_modes'.format(condition))
        save_fig(fig2, 'figure4', 'blu_{}_singular_values'.format(condition))
        save_fig(fig3, 'figure4', 'blu_{}_prediction_error'.format(condition))
