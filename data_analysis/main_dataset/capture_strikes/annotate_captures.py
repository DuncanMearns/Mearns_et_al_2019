from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.video import Video, video_code_to_timestamp
import pandas as pd
import numpy as np
import os
import sys


if __name__ == "__main__":

    capture_strike_info = pd.read_csv(paths['capture_strikes'],
                                      index_col='bout_index',
                                      dtype={'ID': str, 'video_code': str})

    random_examples = np.random.choice(capture_strike_info.index, 100, replace=False)
    random_examples = capture_strike_info.loc[random_examples]

    strike_frames = []
    index_order = []
    for fish_ID, fish_strikes in random_examples.groupby('ID'):
        fish_info = experiment.data[experiment.data['ID'] == fish_ID].iloc[0]
        date_folder = fish_info.date.replace('-', '_')
        for video_code, video_strikes in fish_strikes.groupby('video_code'):
            video_file = video_code_to_timestamp(video_code) + '.avi'
            video_path = os.path.join(experiment.video_directory, date_folder, fish_info['name'], video_file)
            v = Video(video_path)
            for idx, bout_info in video_strikes.iterrows():
                v.scroll(first_frame=bout_info.start, last_frame=bout_info.end)
                if v.enter():
                    strike_frames.append(v.frame_number)
                elif v.space():
                    strike_frames.append(np.nan)
                else:
                    sys.exit()
                index_order.append(idx)

    reordered = random_examples.loc[index_order]
    reordered['strike_frame'] = strike_frames
    reordered = reordered.sort_index()
    reordered.to_csv(paths['strike_frames'])
