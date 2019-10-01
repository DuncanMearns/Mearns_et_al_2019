from paths import paths
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


if __name__ == "__main__":

    capture_strike_info = pd.read_csv(paths['capture_strikes'],
                                      index_col='bout_index',
                                      dtype={'ID': str, 'video_code': str})
    isomap = np.load(paths['isomapped_strikes'])
    assert len(capture_strike_info) == len(isomap)

    cluster_labels = KMeans(2, random_state=1511).fit_predict(isomap)
    capture_strike_info['strike_cluster'] = cluster_labels

    capture_strike_info.to_csv(paths['capture_strikes'])
