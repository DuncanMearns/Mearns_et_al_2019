from plotting import *
from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from datasets.ath5_ablation import experiment as ath5
import numpy as np


transition_directory = os.path.join(blu.parent.subdirs['analysis'], 'transitions')
U, S, Vs_ctrl = np.load(os.path.join(transition_directory, 'USVs.npy'))
U, S, Va_ctrl = np.load(os.path.join(transition_directory, 'USVa.npy'))


for name, experiment, group1, group2 in zip(('blu', 'lak', 'ath5'),
                                            (blu, lak, ath5),
                                            ('het', 'ctrl', 'control'),
                                            ('mut', 'mut', 'ablated')):

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
    U, S, Vs_het = np.load(os.path.join(transition_directory, group1, 'USVs.npy'))
    U, S, Va_het = np.load(os.path.join(transition_directory, group1, 'USVa.npy'))
    U, S, Vs_mut = np.load(os.path.join(transition_directory, group2, 'USVs.npy'))
    U, S, Va_mut = np.load(os.path.join(transition_directory, group2, 'USVa.npy'))

    # Control vs het
    CHs = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            CHs[i, j] = np.abs(np.dot(Vs_ctrl[:, i], Vs_het[:, j]))

    CHa = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            CHa[i, j] = np.abs(np.dot(Va_ctrl[:, i], Va_het[:, j]))

    # Het vs mut
    HMs = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            HMs[i, j] = np.abs(np.dot(Vs_het[:, i], Vs_mut[:, j]))

    HMa = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            HMa[i, j] = np.abs(np.dot(Va_het[:, i], Va_mut[:, j]))

    fig, axes = plt.subplots(2, 2, figsize=(2, 2))
    axes[0][0].matshow(CHs, cmap='inferno', vmin=0, vmax=1)
    axes[0][1].matshow(CHa, cmap='inferno', vmin=0, vmax=1)
    axes[1][0].matshow(HMs, cmap='inferno', vmin=0, vmax=1)
    axes[1][1].matshow(HMa, cmap='inferno', vmin=0, vmax=1)
    for row in axes:
        row[0].set_xticks(np.arange(0, 4))
        row[0].set_xticklabels(np.arange(1, 5))
        row[0].set_yticks(np.arange(0, 4))
        row[0].set_yticklabels(np.arange(1, 5))
        row[1].set_xticks(np.arange(0.5, 8.5, 2))
        row[1].set_xticklabels(np.arange(1, 5))
        row[1].set_yticks(np.arange(0.5, 8.5, 2))
        row[1].set_yticklabels(np.arange(1, 5))

    # save_fig(fig, 'figure4', '{}_dot_products'.format(name))
    plt.close(fig)

    print name
    print HMs[[0, 1], [0, 1]]
    print HMa[:2, :2]
