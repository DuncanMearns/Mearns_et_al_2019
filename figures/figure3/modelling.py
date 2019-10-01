from plotting import *
from datasets.main_dataset import experiment
from plotting.colors import cluster_colors
from scipy import stats as ss
import numpy as np


modelling_directory = os.path.join(experiment.subdirs['analysis'], 'modelling')

p_values = np.empty((7, 5))
statistics = np.empty((7, 5))

fig, axes = plt.subplots(2, 1, figsize=(3, 3))
xticks = [[], []]

for i, l in enumerate([5, 1, 6, 0, 3, 4, 2]):

    sequences, probabilities = np.load(os.path.join(modelling_directory, 'prediction_probabilities_{}.npy'.format(l)))
    probabilities = probabilities[:, ::-1]
    improvement = np.diff(probabilities, axis=1) / probabilities[:, :-1]

    mean_probabilities = np.nanmean(probabilities, axis=0)
    sem_probabilities = np.nanstd(probabilities, axis=0) / np.sqrt(np.sum(~np.isnan(probabilities), axis=0))

    mean_improvements = np.nanmean(improvement, axis=0)
    sem_improvements = np.nanstd(improvement, axis=0) / np.sqrt(np.sum(~np.isnan(improvement), axis=0))

    for j, model1 in enumerate(probabilities.T[:-1]):
        model2 = probabilities.T[j+1]
        stat, pvalue = ss.ttest_ind(model1[~np.isnan(model1)], model2[~np.isnan(model2)])
        p_values[i, j] = pvalue
        statistics[i, j] = stat

    means = [mean_probabilities, mean_improvements]
    errors = [sem_probabilities, sem_improvements]
    n = 0
    for ax, mean, errs in zip(axes, means, errors):
        width = 0.1
        fc = cluster_colors[l].copy()
        fc[-1] = 0.5
        x = i + np.linspace(0, 0.7, len(mean))
        ax.plot(x, mean, c=cluster_colors[l], lw=0.5)
        ax.scatter(x, mean, c=cluster_colors[l], lw=0, s=3)
        ax.fill_between(x, mean - errs, mean + errs, facecolor=cluster_colors[l], alpha=0.5)
        xticks[n].extend(x)
        n += 1

    print mean_improvements

axes[0].set_xticks(xticks[0])
axes[1].set_xticks(xticks[1])
for ax in axes:
    open_ax(ax)
    ax.set_xticklabels([])

axes[0].set_ylim(0, 0.35)
yticks = list(np.arange(0, 0.4, 0.1))
yticks[0] = 0
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(yticks, fontproperties=verysmallfont)
axes[0].set_yticks(np.arange(0, 0.36, 0.02), minor=True)

axes[1].set_ylim(-0.5, 1.5)
yticks = list(np.arange(-0.5, 2, 0.5))
yticks[1] = 0
axes[1].set_yticks(yticks)
axes[1].set_yticklabels(yticks, fontproperties=verysmallfont)
axes[1].set_yticks(np.arange(-0.5, 1.5, 0.1), minor=True)

# print ((statistics < 0) & ((p_values * 35) < 0.01))
print (p_values * 35) < 0.01

plt.show()
# save_fig(fig, 'figure3', 'simplex_projection')
