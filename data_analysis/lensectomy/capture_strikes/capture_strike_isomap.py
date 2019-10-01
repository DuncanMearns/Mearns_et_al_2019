from datasets.lensectomy import experiment
from datasets.main_dataset import experiment as mapping_experiment
from paths import paths
from behaviour_analysis.analysis import BoutData, calculate_distance_matrix, IsomapPrecomputed
from behaviour_analysis.miscellaneous import print_heading, Timer
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


if __name__ == "__main__":

    capture_strikes = pd.read_csv(paths['capture_strikes'], index_col=0, dtype={'ID': str, 'video_code': str})

    mean_tail, std_tail = np.load(os.path.join(mapping_experiment.subdirs['analysis'],
                                               'behaviour_space',
                                               'tail_statistics.npy'))
    eigenfish = np.load(os.path.join(mapping_experiment.subdirs['analysis'],
                                     'behaviour_space',
                                     'eigenfish.npy'))

    # Import bouts
    capture_strike_data = BoutData.from_directory(capture_strikes, experiment.subdirs['kinematics'],
                                                  check_tail_lengths=False, tail_columns_only=True)

    # Transform
    transformed_strikes = capture_strike_data.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)
    transformed_strikes = transformed_strikes.list_bouts(values=True, ndims=3)

    # Truncate
    truncated_strikes = [bout[12:37] for bout in transformed_strikes]
    bw = 0.006  # 3 frames

    # Calculate distance matrix
    print_heading('CALCULATING CAPTURE STRIKE DISTANCE MATRIX')
    timer = Timer()
    timer.start()
    D_normal = calculate_distance_matrix(truncated_strikes, bw=bw)
    D_flipped = calculate_distance_matrix(truncated_strikes, bw=bw, flip=True)
    D = np.min([D_normal, D_flipped], axis=0)
    np.save(paths['distance_matrix'], D)
    time_taken = timer.stop()
    print 'Time taken: {}'.format(timer.convert_time(time_taken))

    # Perform embedding
    D = squareform(D)
    np.random.seed(1992)
    isomap = IsomapPrecomputed(n_neighbors=5, n_components=2)
    isomapped_strikes = isomap.fit_transform(D)
    np.save(paths['isomapped_strikes'], isomapped_strikes)

    # Cluster
    cluster_labels = KMeans(2).fit_predict(isomapped_strikes)
    capture_strikes['cluster_label'] = cluster_labels
    capture_strikes.to_csv(paths['capture_strikes'], index=True)
