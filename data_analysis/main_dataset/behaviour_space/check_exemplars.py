from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.video import Video, video_code_to_timestamp
import pandas as pd
import os


if __name__ == "__main__":

    exemplars = pd.read_csv(paths['exemplars'], dtype={'ID': str, 'video_code': str})
    to_check = exemplars[pd.isnull(exemplars['clean'])]

    for idx, bout_info in to_check.iterrows():
        if idx % 100 == 0:
            print 'Analysed: {}/{}'.format(idx, len(exemplars))
        fish_info = experiment.data[experiment.data['ID'] == bout_info.ID].iloc[0]
        date_folder = fish_info.date.replace('-', '_')
        video_file = video_code_to_timestamp(bout_info.video_code) + '.avi'
        video_path = os.path.join(experiment.video_directory, date_folder, fish_info['name'], video_file)
        v = Video(video_path)
        v.scroll(first_frame=bout_info.start, last_frame=bout_info.end, name='exemplar_' + str(idx))
        if v.enter():
            exemplars.loc[idx, 'clean'] = True
        elif v.space():
            exemplars.loc[idx, 'clean'] = False
        else:
            break

    exemplars.to_csv(paths['exemplars'], index=False)
