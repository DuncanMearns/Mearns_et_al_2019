from datasets.main_dataset import experiment
import numpy as np
import os
from scipy.stats import spearmanr


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
    USVs = np.load(os.path.join(transition_directory, 'USVs.npy'))
    USVa = np.load(os.path.join(transition_directory, 'USVa.npy'))

    eye_convergence_directory = os.path.join(experiment.subdirs['analysis'], 'eye_convergence')
    ec_scores = np.load(os.path.join(eye_convergence_directory, 'state_convergence_scores.npy'))

    pci = ec_scores[:, 1:].sum(axis=1) - ec_scores[:, 0]
    S1 = USVs[2, :, 1]
    print 'PCI vs. S1:', spearmanr(pci, S1)

    A1_0 = USVa[2, :, 0]
    A1_1 = USVa[2, :, 1]

    print 'Early:', spearmanr(A1_0, ec_scores[:, 1])
    print 'Mid:', spearmanr(A1_1, ec_scores[:, 2])
    print 'Late:', spearmanr(A1_0, ec_scores[:, 3])
