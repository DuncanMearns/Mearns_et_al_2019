from plotting import *
from datasets.main_dataset import experiment
import numpy as np
from scipy.stats import ks_2samp


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
    sym = np.load(os.path.join(transition_directory, 'symmetric_models.npy'))  # (n_permutations, n_models)
    asym = np.load(os.path.join(transition_directory, 'antisymmetric_models.npy'))  # (n_permutation, n_models)

    asym = np.concatenate([sym[:, [0]], asym], axis=1)  # Add the null model to the antisymmetric models

    # Plot percent improvement in prediction
    fig, axes = plt.subplots(1, 2, figsize=(2, 1))
    for ax, models in zip(axes, [sym, asym]):
        improvement = 100 * (models[:, [0]] - models[:, 1:]) / models[:, [0]]
        mean_improvement = np.mean(improvement, axis=0)
        std_improvement = np.std(improvement, axis=0)
        ax.scatter(np.arange(1, 6), mean_improvement, s=10, lw=1, c='w', edgecolor='k', zorder=1)
        ax.errorbar(np.arange(1, 6), mean_improvement, yerr=std_improvement,
                    capsize=3, fmt='none', zorder=0, c='k', lw=1)
        open_ax(ax)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(np.arange(1, 6))
        ax.set_xticklabels(np.arange(1, 6), fontproperties=verysmallfont)
        print zip(mean_improvement, std_improvement)

    axes[0].set_ylim(-10, 40)
    yticks = list(np.arange(-10, 50, 10))
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticks, fontproperties=verysmallfont)
    axes[0].set_yticks(np.arange(-10, 40, 2), minor=True)

    axes[1].set_ylim(-2, 3)
    yticks = list(np.arange(-2, 4, 1))
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(yticks, fontproperties=verysmallfont)
    axes[1].set_yticks(np.arange(-2, 3, 0.2), minor=True)

    # plt.show()
    # save_fig(fig, 'figure2', 'svd_model_prediction')

    # Stats
    print 'Symmetric models'
    for model in sym[:, 1:].T:
        print ks_2samp(sym[:, 0], model)
    print '\nAnti-symmetric models'
    for model in asym.T:
        print ks_2samp(sym[:, 0], model)
