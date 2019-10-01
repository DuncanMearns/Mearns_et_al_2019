from plotting import *
from datasets.spontaneous import experiment
import numpy as np


prey_capture_directory = os.path.join(experiment.parent.subdirs['analysis'], 'transitions')
U, S, Vs_pc = np.load(os.path.join(prey_capture_directory, 'USVs.npy'))
U, S, Va_pc = np.load(os.path.join(prey_capture_directory, 'USVa.npy'))


spontaneous_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
U, S, Vs_spon = np.load(os.path.join(spontaneous_directory, 'USVs.npy'))
U, S, Va_spon = np.load(os.path.join(spontaneous_directory, 'USVa.npy'))


# Control vs het
CHs = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        CHs[i, j] = np.abs(np.dot(Vs_pc[:, i], Vs_spon[:, j]))

CHa = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        CHa[i, j] = np.abs(np.dot(Va_pc[:, i], Va_spon[:, j]))


fig, axes = plt.subplots(1, 2, figsize=(2, 1))
axes[0].matshow(CHs, cmap='inferno', vmin=0, vmax=1)
axes[1].matshow(CHa, cmap='inferno', vmin=0, vmax=1)

axes[0].set_xticks(np.arange(0, 4))
axes[0].set_xticklabels(np.arange(1, 5))
axes[0].set_yticks(np.arange(0, 4))
axes[0].set_yticklabels(np.arange(1, 5))
axes[1].set_xticks(np.arange(0.5, 8.5, 2))
axes[1].set_xticklabels(np.arange(1, 5))
axes[1].set_yticks(np.arange(0.5, 8.5, 2))
axes[1].set_yticklabels(np.arange(1, 5))

# plt.show()
save_fig(fig, 'figureS3', 'compare_transition_modes')
