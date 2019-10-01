from plotting import *
from datasets.main_dataset import experiment

import numpy as np


transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
sv = np.load(os.path.join(transition_directory, 'singular_value_permutation.npy'))


# Plot the singular values
fig, axes = plt.subplots(1, 2, figsize=(2, 1))

# Real symmetric
real_svs_mean = sv[:, 0, :].mean(axis=0)
real_svs_std = sv[:, 0, :].std(axis=0)
axes[0].fill_between(np.arange(0, 10), real_svs_mean - real_svs_std, real_svs_mean + real_svs_std,
                     facecolor='r', alpha=0.5)

# Shuffled symmetric
shuf_svs_mean = sv[:, 2, :].mean(axis=0)
shuf_svs_std = sv[:, 2, :].std(axis=0)
axes[0].fill_between(np.arange(0, 10), shuf_svs_mean - shuf_svs_std, shuf_svs_mean + shuf_svs_std,
                     facecolor='k', alpha=0.5)

# Real antisymmetric
real_sva_mean = sv[:, 1, :].mean(axis=0)
real_sva_std = sv[:, 1, :].std(axis=0)
axes[1].fill_between(np.arange(1, 11), real_sva_mean - real_sva_std, real_sva_mean + real_sva_std,
                     facecolor='r', alpha=0.5)

# Shuffled antismmetric
shuf_sva_mean = sv[:, 3, :].mean(axis=0)
shuf_sva_std = sv[:, 3, :].std(axis=0)
axes[1].fill_between(np.arange(1, 11), shuf_sva_mean - shuf_sva_std, shuf_sva_mean + shuf_sva_std,
                     facecolor='k', alpha=0.5)

for ax in axes:
    open_ax(ax)

axes[0].set_ylim(0, 25)
yticks = np.arange(0, 30, 5)
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(yticks, fontproperties=verysmallfont)
axes[0].set_yticks(np.arange(25), minor=True)

axes[0].set_xlim(-0.5, 9)
axes[0].set_xticks(np.arange(1, 10, 2))
axes[0].set_xticklabels(np.arange(1, 10, 2), fontproperties=verysmallfont)

axes[1].set_ylim(0, 2.5)
yticks = list(np.arange(0, 3.0, 0.5))
yticks[0] = 0
axes[1].set_yticks(yticks)
axes[1].set_yticklabels(yticks, fontproperties=verysmallfont)
axes[1].set_yticks(np.arange(0, 2.5, 0.1), minor=True)

axes[1].set_xlim(0.5, 10)
axes[1].set_xticks(np.arange(1, 10, 2))
axes[1].set_xticklabels(np.arange(1, 10, 2), fontproperties=verysmallfont)

save_fig(fig, 'figure2', 'singular_values')
# plt.show()
