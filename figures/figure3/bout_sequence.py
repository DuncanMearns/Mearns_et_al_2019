from plotting import *
from plotting.colors import cluster_colors
from plotting.plots.tail import generate_reconstructed_points
from datasets.main_dataset.example_data import kinematics, tail_columns, example_bouts
import numpy as np

ps = generate_reconstructed_points(kinematics.loc[:, tail_columns].values, 0)
tip_angle = np.degrees(np.arcsin(ps[:, 1, -1] / 50.))

fig, ax = plt.subplots(figsize=(1.5, 0.5))

ax.plot(kinematics.index / 500., tip_angle, c='k', lw=0.5)
for idx, bout_info in example_bouts.iterrows():
    t_ = np.arange(bout_info.start, bout_info.end + 1) / 500.
    color = cluster_colors[bout_info['module']]
    ax.fill_between(t_, np.ones((len(t_),)) * (-40), np.ones((len(t_),)) * 20, facecolor=color, alpha=0.8)
ax.axis('off')

# plt.show()
save_fig(fig, 'figure3', 'annotated_sequence')
