from datasets.main_dataset import experiment
from behaviour_analysis.analysis.transitions import SVD
import numpy as np
import os

if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')
    WTW = np.load(os.path.join(transition_directory, 'WTW.npy'))
    USVs, USVa = SVD(WTW)
    np.save(os.path.join(transition_directory, 'USVs.npy'), USVs)
    np.save(os.path.join(transition_directory, 'USVa.npy'), USVa)
