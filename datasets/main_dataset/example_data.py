from . import experiment
from ast import literal_eval
import os
import numpy as np
import pandas as pd
from behaviour_analysis.miscellaneous import read_csv
from behaviour_analysis.video import Video

# Example data
fish_ID = '2017081006'
timestamp = '19-30-58'
first_frame, last_frame = 19300, 22300
example_frame = 21112

fish_data = experiment.data[experiment.data['ID'] == fish_ID].iloc[0]
video_code = fish_ID + timestamp.replace('-', '')

# Get tracking data
tracking_path = os.path.join(experiment.directory, fish_data.tracking_directory, video_code + '.csv')
tracking = read_csv(tracking_path, centre=literal_eval, right_centre=literal_eval, left_centre=literal_eval)

# Get tail points
points_path = os.path.join(experiment.directory, fish_data.tracking_directory, video_code + '.npy')
points = np.load(points_path)

# Get kinematic data
kinematics_path = os.path.join(experiment.directory, fish_data.kinematics_directory, video_code + '.csv')
kinematics = pd.read_csv(kinematics_path).loc[first_frame:last_frame]
tail_columns = [col for col in kinematics.columns if col[0] == 'k']

# Get bouts
bouts_path = os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv')
bouts_df = pd.read_csv(bouts_path, index_col='bout_index', dtype={'ID': str, 'video_code': str})
video_bouts = bouts_df.groupby('video_code').get_group(video_code)
example_bouts = video_bouts[(video_bouts['start'] >= first_frame) & (video_bouts['end'] <= last_frame)]

# Set video path
video_path = os.path.join(experiment.video_directory, fish_data.video_directory, timestamp + '.avi')
video = Video(video_path)

# Get convergence threshold
convergence_scores = pd.read_csv(os.path.join(experiment.subdirs['analysis'],
                                              'eye_convergence',
                                              'convergence_scores.csv'), dtype={'ID': str})
convergence_threshold = convergence_scores[convergence_scores['ID'] == fish_ID]['threshold'].iloc[0]

data = dict(video=video,
            tracking=tracking,
            kinematics=kinematics,
            points=points,
            example_frame=example_frame,
            first_frame=first_frame,
            last_frame=last_frame,
            example_bouts=example_bouts,
            convergence_threshold=convergence_threshold)
