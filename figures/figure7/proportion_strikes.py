from plotting import *
from plotting.colors import lensectomy_colors
from plotting.plots import boxplot
from datasets.lensectomy import experiment
import numpy as np
import os


capture_strike_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes')

ctrl_proportions = np.load(os.path.join(capture_strike_directory, 'proportion_strikes', 'control_proportions.npy'))
uni_proportions = np.load(os.path.join(capture_strike_directory, 'proportion_strikes', 'unilateral_proportions.npy'))
bi_proportions = np.load(os.path.join(capture_strike_directory, 'proportion_strikes', 'bilateral_proportions.npy'))

fig, ax = plt.subplots(figsize=(1, 1.5))
group_rates = [ctrl_proportions, uni_proportions, bi_proportions]
groups = ('control', 'unilateral', 'bilateral')
boxplot(group_rates, group_colors=[lensectomy_colors[group] for group in groups])

ax.set_xticks([])

ax.set_ylim(0, 0.4)
yticks = list(np.arange(0, 0.5, 0.1))
yticks[0] = 0
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontproperties=verysmallfont)
ax.set_yticks(np.arange(0, 0.4, 0.02), minor=True)

save_fig(fig, 'figure7', 'proportion_strikes')
# plt.show()
