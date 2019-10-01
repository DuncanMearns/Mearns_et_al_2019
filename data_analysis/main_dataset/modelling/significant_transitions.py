from datasets.main_dataset import experiment
from behaviour_analysis.analysis.transitions import fish_transition_matrices
from behaviour_analysis.statistics import bonferroni_holm
import numpy as np
import pandas as pd
from scipy import stats as ss
import os

modelling_directory = os.path.join(experiment.subdirs['analysis'], 'modelling')

if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col='transition_index', dtype={'ID': str, 'video_code': str})
    n_clusters = len(mapped_bouts['module'].unique())

    S = []
    for shuffle in np.arange(1000):
        if shuffle % 10 == 0:
            print shuffle
        shuffled = fish_transition_matrices(mapped_bouts, state_col='module', shuffle=True, verbose=False)
        S.append(shuffled.sum(axis=0))
    S = np.array(S)

    T = fish_transition_matrices(mapped_bouts, state_col='module', shuffle=False)
    T = T.sum(axis=0)

    model_params = np.array([S.mean(axis=0), S.std(axis=0)])
    p_values_decrease = np.zeros((n_clusters, n_clusters))

    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            norm = ss.norm(loc=model_params[0, i, j], scale=model_params[1, i, j])
            p_values_decrease[i, j] = norm.cdf(T[i, j])
    p_values_increase = 1 - p_values_decrease

    significant_decrease = bonferroni_holm(p_values_decrease, 0.05)
    significane_increase = bonferroni_holm(p_values_increase, 0.05)
    print significane_increase

    np.save(os.path.join(modelling_directory, 'T_clusters.npy'), T)
    np.save(os.path.join(modelling_directory, 'S_clusters.npy'), S)
    np.save(os.path.join(modelling_directory, 'S_params.npy'), model_params)
    np.save(os.path.join(modelling_directory, 'p-+.npy'), np.array([p_values_decrease,
                                                                    p_values_increase,
                                                                    significant_decrease,
                                                                    significane_increase]))
