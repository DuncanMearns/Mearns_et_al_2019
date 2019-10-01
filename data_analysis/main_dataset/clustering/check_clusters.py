from setup import *
from behaviour_analysis.video import Video, video_code_to_timestamp
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":

    exemplars = pd.read_csv(paths['exemplars'], index_col='bout_index', dtype={'ID': str, 'video_code': str})
    exemplars = exemplars[exemplars['clean']]

    joe_labels = np.load(os.path.join(experiment.subdirs['analysis'], 'clustering', 'joe_labels.npy'))
    exemplars['module'] = joe_labels

    for module, module_bouts in exemplars.groupby('module'):
        random_examples = module_bouts.loc[np.random.choice(module_bouts.index, 10)]
        for idx, bout_info in random_examples.iterrows():
            fish_info = experiment.data[experiment.data['ID'] == bout_info.ID].iloc[0]
            video_file = video_code_to_timestamp(bout_info.video_code)
            video_path = os.path.join(experiment.video_directory, fish_info.video_directory, video_file + '.avi')
            v = Video(video_path)
            v.play(first_frame=bout_info.start, last_frame=bout_info.end, name='cluster ' + str(bout_info.module))
