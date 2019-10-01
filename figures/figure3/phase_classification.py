from plotting import *
from plotting.colors import cluster_colors
from datasets.main_dataset import experiment
import pandas as pd
import numpy as np

mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                           index_col='bout_index', dtype={'ID': str, 'video_code': str})

labels = mapped_bouts['module']
phases = mapped_bouts['phase']
n_clusters = len(labels.unique())

# ---------------
# Module by phase
# ---------------

rcParams['hatch.color'] = 'firebrick'

fig1, axes = plt.subplots(n_clusters, 1, figsize=(1, n_clusters))

for l, ax in enumerate(axes):
    is_module = (labels == l)
    n = float(is_module.sum())
    proportions = []
    for phase in range(4):
        proportions.append((phases == phase)[is_module].sum() / n)
    patches, _ = ax.pie(proportions, np.array([0.1, 0, 0, 0]),
                        colors=['skyblue', 'lightsalmon', 'lightsalmon', 'firebrick'], startangle=90)
    patches[2].set_hatch('.' * 6)
    ax.axis('equal')
# plt.show()
save_fig(fig1, 'figureS4', 'clusters_by_phase')

# ---------------
# Phase by module
# ---------------

fig2, axes = plt.subplots(4, 1, figsize=(1, 4))

for phase, ax in enumerate(axes):
    is_phase = (phases == phase)
    n = float(is_phase.sum())
    proportions = [is_phase[labels == l].sum() / n for l in range(n_clusters)]
    ax.pie(proportions, colors=cluster_colors, startangle=90)
    ax.axis('equal')
# plt.show()
save_fig(fig2, 'figureS4', 'phases_by_cluster')

# --------------------
# Total bout by module
# --------------------

fig3, ax = plt.subplots(figsize=(1, 1))
proportions = [(labels == l).sum() / float(len(labels)) for l in range(n_clusters)]
ax.pie(proportions, colors=cluster_colors, startangle=90)
# plt.show()
save_fig(fig3, 'figureS4', 'total_by_cluster')
