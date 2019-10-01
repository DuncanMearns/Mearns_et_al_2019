from plotting import *
from datasets.lensectomy import experiment
import numpy as np


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
    U, S, Vs_ctrl = np.load(os.path.join(transition_directory, 'control', 'USVs.npy'))
    U, S, Va_ctrl = np.load(os.path.join(transition_directory, 'control', 'USVa.npy'))
    U, S, Vs_uni = np.load(os.path.join(transition_directory, 'unilateral', 'USVs.npy'))
    U, S, Va_uni = np.load(os.path.join(transition_directory, 'unilateral', 'USVa.npy'))

    Va_ctrl[:, :2] = Va_ctrl[:, (1, 0)]

    S = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            S[i, j] = np.abs(np.dot(Vs_ctrl[:, i], Vs_uni[:, j]))

    A = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            A[i, j] = np.abs(np.dot(Va_ctrl[:, i], Va_uni[:, j]))

    fig, axes = plt.subplots(2, 1, figsize=(1, 2))
    axes[0].matshow(S, cmap='inferno', vmin=0, vmax=1)
    axes[1].matshow(A, cmap='inferno', vmin=0, vmax=1)

    axes[0].set_xticks(np.arange(0, 4))
    axes[0].set_xticklabels(np.arange(1, 5))
    axes[0].set_yticks(np.arange(0, 4))
    axes[0].set_yticklabels(np.arange(1, 5))
    axes[1].set_xticks(np.arange(0.5, 8.5, 2))
    axes[1].set_xticklabels(np.arange(1, 5))
    axes[1].set_yticks(np.arange(0.5, 8.5, 2))
    axes[1].set_yticklabels(np.arange(1, 5))

    plt.show()
