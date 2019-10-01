from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.video import Video, video_code_to_timestamp
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":

    capture_strike_info = pd.read_csv(paths['capture_strikes'],
                                      index_col='bout_index',
                                      dtype={'ID': str, 'video_code': str})

    by_cluster = capture_strike_info.groupby('strike_cluster')  # 0 for attack; 1 for s-strike
    for label, cluster_bouts in by_cluster:
        random_examples = cluster_bouts.loc[np.random.choice(cluster_bouts.index, 20, replace=False)]
        for idx, bout_info in random_examples.iterrows():
            fish_info = experiment.data[experiment.data['ID'] == bout_info.ID].iloc[0]
            video_file = video_code_to_timestamp(bout_info.video_code)
            video_path = os.path.join(experiment.video_directory, fish_info.video_directory, video_file + '.avi')
            v = Video(video_path)
            v.play(first_frame=bout_info.start, last_frame=bout_info.end, name='cluster ' + str(label))
