from .image_stimulus_map import image_stimulus_map
from ...video import Video, video_code_to_timestamp
from ...tracking import background_division
from ...miscellaneous import read_csv, Timer
import cv2
import numpy as np
from ast import literal_eval
import os


def calculate_fish_sequences(ID, bouts, experiment, output_directory):
    """Calculate stimulus sequences for bouts belonging to a given fish"""
    fish_bouts = bouts.groupby('ID').get_group(ID)
    output_path = os.path.join(output_directory, ID + '.npy')
    mapper = StimulusSequenceMapper(ID, experiment, fish_bouts, output_path)
    time_taken = mapper.run()
    return time_taken


class StimulusSequenceMapper():
    """Calculate a stimulus sequence for bouts belonging to a single fish"""

    columns = ['left_centre', 'right_centre', 'heading']

    def __init__(self, ID, experiment, bouts, output_path):
        self.ID = ID
        self.experiment = experiment
        self.bouts = bouts
        self.output_path = output_path

    def run(self):
        # Calculate stimulus maps for fish
        timer = Timer()
        timer.start()
        # Fish info
        fish_info = self.experiment.data[self.experiment.data['ID'] == self.ID].iloc[0]
        tracking_directory = os.path.join(self.experiment.directory, fish_info.tracking_directory)
        background_path = os.path.join(self.experiment.directory, fish_info.background_path)
        background = cv2.imread(background_path, 0)
        # Create space for sequences in RAM
        fish_sequences = np.zeros((len(self.bouts), 500, background.shape[0] / 2, background.shape[1] / 2),
                                  dtype='uint8')
        i = 0
        for video_code, video_bouts in self.bouts.groupby('video_code'):
            video_file = video_code_to_timestamp(video_code)
            video_path = os.path.join(self.experiment.video_directory,
                                      fish_info.video_directory,
                                      video_file + '.avi')
            tracking_path = os.path.join(tracking_directory, video_code + '.csv')
            video = Video(video_path)
            tracking = read_csv(tracking_path, **dict(zip(self.columns, [literal_eval] * 2 + [float])))
            for idx, bout_info in video_bouts.iterrows():
                first_frame, last_frame = bout_info.start - 450, bout_info.start + 50
                frames = video.return_frames(first_frame, last_frame)
                frames = np.asarray(frames)[..., 0]
                sub = background_division(frames, background)
                j = 0
                for f, img in zip(np.arange(first_frame, last_frame), sub):
                    tracking_info = tracking.loc[f, self.columns].to_dict()
                    stim_map = image_stimulus_map(img, tracking_info)
                    fish_sequences[i, j] = stim_map
                    j += 1
                i += 1
        np.save(self.output_path, fish_sequences)
        time_taken = timer.stop()
        return time_taken
