from . import experiment
from behaviour_analysis.video import Video
from behaviour_analysis.miscellaneous import read_csv
import pandas as pd
import numpy as np
from ast import literal_eval
import os


bouts_path = os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv')
bouts_df = pd.read_csv(bouts_path, dtype={'ID': str, 'video_code': str})

jaw_path = os.path.join(experiment.subdirs['analysis'], 'mapped_jaw.csv')
jaw_df = pd.read_csv(jaw_path, dtype={'ID': str, 'video_code': str}, index_col=0)

fish_ID = '2017060204'
timestamp = '18-36-19'
first_frame, last_frame = 0, 1600
example_frame = 700

fish_data = experiment.data[experiment.data['ID'] == fish_ID].iloc[0]
video_code = fish_ID + timestamp.replace('-', '')

# Get tracking data
tracking_path = os.path.join(experiment.directory, fish_data.tracking_directory, video_code + '.csv')
tracking = read_csv(tracking_path, centre=literal_eval, right_centre=literal_eval, left_centre=literal_eval,
                    side_centre=literal_eval, head_midpoint=literal_eval, hyoid=literal_eval)

# Get tail points
points_path = os.path.join(experiment.directory, fish_data.tracking_directory, video_code + '.npy')
points = np.load(points_path)

# Get kinematic data
kinematics_path = os.path.join(experiment.directory, fish_data.kinematics_directory, video_code + '.csv')
kinematics = pd.read_csv(kinematics_path).loc[first_frame:last_frame]

# Get bouts
video_bouts = bouts_df.groupby('video_code').get_group(video_code)
example_bouts = video_bouts[(video_bouts['start'] >= first_frame) & (video_bouts['end'] <= last_frame)]

# Get jaw movements
video_jaw = jaw_df.groupby('video_code').get_group(video_code)
example_jaw = video_jaw[(video_jaw['start'] >= first_frame) & (video_jaw['end'] <= last_frame)]

# Set video path
video_path = os.path.join(experiment.video_directory, fish_data.video_directory, timestamp + '.avi')
video = Video(video_path)

# ROIs
top_ROI = fish_data.top_ROI
side_ROI = fish_data.side_ROI

data = dict(video=video, tracking=tracking, kinematics=kinematics, points=points,
            first_frame=first_frame, last_frame=last_frame, example_bouts=example_bouts,
            top_ROI=top_ROI, side_ROI=side_ROI)
