from plotting import *
from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
import numpy as np
from scipy.cluster import hierarchy as sch
import os


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions_old')
    WTW = np.load(os.path.join(transition_directory, 'WTW.npy'))
    USVs = np.load(os.path.join(transition_directory, 'USVs.npy'))
    USVa = np.load(os.path.join(transition_directory, 'USVa.npy'))

    S = 0.5 * (WTW + WTW.T)
    A = 0.5 * (WTW - WTW.T)

    # All transitions
    q, r = np.linalg.qr(np.concatenate([USVs[0, :, 1:3],
                                        USVa[0, :, :2]], axis=1))
    Y = sch.linkage(q, method='ward')
    Z1 = sch.dendrogram(Y, orientation='left', no_plot=True)
    Z2 = sch.dendrogram(Y, no_plot=True)
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    T_sorted = WTW[np.meshgrid(idx1, idx2)]
    S_sorted = S[np.meshgrid(idx1, idx2)]
    A_sorted = A[np.meshgrid(idx1, idx2)]

    fig2_dir = create_folder(output_directory, 'figure2')
    matrix_dir = create_folder(fig2_dir, 'matrices')

    # Transition matrix
    fig, ax = plt.subplots(figsize=(1.5, 1.5), gridspec_kw=dict(left=0, right=1, bottom=0, top=1))
    ax.matshow(T_sorted, cmap='plasma', vmin=0, vmax=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(matrix_dir, 'T.png'))
    plt.close(fig)

    # Symmetric matrix
    fig, ax = plt.subplots(figsize=(1.5, 1.5), gridspec_kw=dict(left=0, right=1, bottom=0, top=1))
    ax.matshow(S_sorted, cmap='plasma', vmin=-0.1, vmax=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(matrix_dir, 'S.png'))
    plt.close(fig)

    # Antisymmetric matrix
    fig, ax = plt.subplots(figsize=(1.5, 1.5), gridspec_kw=dict(left=0, right=1, bottom=0, top=1))
    ax.matshow(A_sorted, cmap='plasma', vmin=-0.01, vmax=0.01)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(matrix_dir, 'A.png'))
    plt.close(fig)

    # Symmetric modes
    for i in range(3):
        u = USVs[0, :, i]
        s = USVs[1, i, i]
        v = USVs[2, :, i]
        M = np.outer(u, v) * s
        M_sorted = M[np.meshgrid(idx1, idx2)]
        fig, ax = plt.subplots(figsize=(1.5, 1.5), gridspec_kw=dict(left=0, right=1, bottom=0, top=1))
        ax.matshow(M_sorted, cmap='plasma', vmin=-0.1, vmax=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(matrix_dir, 'S_{}.png'.format(i)))
        plt.close(fig)

    # Antisymmetric modes
    for i in range(3):
        M = np.zeros(T_sorted.shape)
        for j in range(2):
            u = USVa[0, :, (2*i)+j]
            s = USVa[1, (2*i)+j, (2*i)+j]
            v = USVa[2, :, (2*i)+j]
            Mj = np.outer(u, v) * s
            M += Mj
        M_sorted = M[np.meshgrid(idx1, idx2)]
        fig, ax = plt.subplots(figsize=(1.5, 1.5), gridspec_kw=dict(left=0, right=1, bottom=0, top=1))
        ax.matshow(M_sorted, cmap='plasma', vmin=-0.01, vmax=0.01)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(matrix_dir, 'A_{}.png'.format(i)))
        plt.close(fig)
    #
    # plt.show()
