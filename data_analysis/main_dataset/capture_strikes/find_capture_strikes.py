from datasets.main_dataset import experiment
from paths import paths
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Import bout info
    dtype = {'ID': str, 'video_code': str}
    bouts_df = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                           index_col='bout_index', dtype=dtype)

    # Import convergence states
    eye_convergence_directory = os.path.join(experiment.subdirs['analysis'], 'eye_convergence')
    normalised_phase_counts = np.load(os.path.join(eye_convergence_directory, 'state_convergence_scores.npy'))

    # Find the capture strike clusters
    late_pc_enriched = normalised_phase_counts[:, 3] > 0.5
    capture_strike_clusters = np.where(late_pc_enriched)[0]
    is_strike_cluster = np.isin(bouts_df['state'], capture_strike_clusters)

    capture_strike_info = bouts_df[is_strike_cluster]
    print len(capture_strike_info)

    capture_strike_info.to_csv(paths['capture_strikes'])
