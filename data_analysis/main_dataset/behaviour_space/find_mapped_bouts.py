from paths import paths
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Last 5 minutes of 2017_08_17 fish1 has bad tracking and bouts should be removed
    bad_ID = '2017081701'
    remove_codes = [bad_ID + '173436',
                    bad_ID + '173536',
                    bad_ID + '173637',
                    bad_ID + '173737',
                    bad_ID + '173837']

    bout_indices = np.load(paths['bout_indices'])
    bouts_df = pd.read_csv(paths['bouts'], dtype={'ID': str, 'video_code': str})
    bouts_df = bouts_df.loc[bout_indices]

    cluster_labels = np.load(paths['cluster_labels'])
    exemplars = pd.read_csv(paths['exemplars'], dtype={'ID': str, 'video_code': str})

    bad_exemplar_idxs = exemplars[exemplars['video_code'].isin(remove_codes)].index
    exemplars.loc[bad_exemplar_idxs, 'clean'] = False

    mapped_clusters = exemplars['cluster'].values
    mapped = np.isin(cluster_labels, mapped_clusters)
    mapped_to_bouts = np.isin(cluster_labels[mapped], mapped_clusters[exemplars['clean']])
    index_shift = np.cumsum(~mapped_to_bouts)

    mapped_bouts = bouts_df[mapped]

    mapped_bouts['transition_index'] = mapped_bouts.index - index_shift
    mapped_bouts = mapped_bouts[mapped_to_bouts]
    mapped_bouts['bout_index'] = mapped_bouts.index
    mapped_bouts = mapped_bouts.set_index('transition_index')

    exemplar_bouts = exemplars[exemplars['clean']]['cluster'].tolist()
    mapped_bouts['state'] = [exemplar_bouts.index(l) for l in cluster_labels[mapped][mapped_to_bouts]]
    mapped_bouts = mapped_bouts[~mapped_bouts['video_code'].isin(remove_codes)]
    assert len(mapped_bouts['state'].unique()) == len(exemplar_bouts)

    mapped_bouts.to_csv(paths['mapped_bouts'], index=True)
    exemplars.to_csv(paths['exemplars'], index=False)
