from datasets.blumenkohl import experiment
from behaviour_analysis.analysis.transitions import CrossValidateSVD
import numpy as np
import os
from scipy.stats import ttest_ind

if __name__ == "__main__":

    transition_directory = os.path.join(experiment.subdirs['analysis'], 'transitions')

    for condition, fish_info in experiment.data.groupby('condition'):

        print condition

        condition_directory = os.path.join(transition_directory, condition)

        # T = np.load(os.path.join(condition_directory, 'smoothed_transition_matrices.npy'))
        #
        # print 'Cross validation'
        # cv = CrossValidateSVD(T, n_permutations=10000, n_sym=5, n_asym=5, n_threads=20)
        # sym, asym = cv.run()
        #
        # np.save(os.path.join(condition_directory, 'symmetric_models.npy'), sym)
        # np.save(os.path.join(condition_directory, 'antisymmetric_models.npy'), asym)

        sym = np.load(os.path.join(condition_directory, 'symmetric_models.npy'))
        asym = np.load(os.path.join(condition_directory, 'antisymmetric_models.npy'))

        print 'Symmetric models'
        for model in sym[:, 1:].T:
            print ttest_ind(sym[:, 0], model)
        print '\nAnti-symmetric models'
        for model in asym.T:
            print ttest_ind(sym[:, 0], model)
        print'\n'
