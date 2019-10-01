from datasets.lensectomy import experiment
from datasets.main_dataset import experiment as mapping_experiment
from paths import paths
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Find capture strike states
    state_convergence = np.load(os.path.join(mapping_experiment.subdirs['analysis'], 'eye_convergence', 'state_convergence_scores.npy'))
    strike_states = np.where(state_convergence[:, 3] > 0.5)[0]

    # Find capture strikes
    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index_col=0,
                               dtype={'ID': str, 'video_code': str})
    end_hunts = mapped_bouts[(mapped_bouts['phase'] == 3) & mapped_bouts['exemplar'].isin(strike_states)]
    end_hunts.to_csv(paths['capture_strikes'], index=True)
