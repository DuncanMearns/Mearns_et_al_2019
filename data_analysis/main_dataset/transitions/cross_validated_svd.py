from behaviour_analysis.analysis.transitions import CrossValidateSVD
import numpy as np
from datasets.main_dataset import experiment
import os


if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')

    T = np.load(os.path.join(transition_directory, 'smoothed_transition_matrices.npy'))

    print 'Cross validation'
    cv = CrossValidateSVD(T, n_permutations=10000, n_sym=5, n_asym=5, n_threads=20)
    sym, asym = cv.run()

    np.save(os.path.join(transition_directory, 'symmetric_models.npy'), sym)
    np.save(os.path.join(transition_directory, 'antisymmetric_models.npy'), asym)
