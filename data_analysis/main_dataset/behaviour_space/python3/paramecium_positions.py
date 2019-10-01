from behaviour_analysis.experiments import TrackingExperiment2D
from behaviour_analysis.manage_files_helpers import create_folder
import os
from behaviour_analysis.multi_threading import MultiThreading
from behaviour_analysis.tracking import background_division, rotate_and_centre_image
from behaviour_analysis.miscellaneous_helpers import read_csv
from behaviour_analysis.video import video_code_to_timestamp
import numpy as np
import cv2
import pandas as pd
from ast import literal_eval
from joblib import Parallel, delayed


class StimulusMap(MultiThreading):

    dtype = dict(zip(['centre', 'left_centre', 'right_centre'], [literal_eval] * 3))
    cols = ['left_centre', 'right_centre', 'heading']

    def __init__(self, video_directory, tracking_directory, fish_bouts, background):
        MultiThreading.__init__(self, 20)
        self.video_directory = video_directory
        self.tracking_directory = tracking_directory
        self.fish_bouts = fish_bouts.groupby('video_code')
        self.background = background
        self.video_codes = fish_bouts['video_code'].unique()
        self.images = {}

    def _run_on_thread(self, arg):
        self.analyse_video(arg)

    def run(self):
        self._run(*self.video_codes)
        all_images = np.concatenate([self.images[video_code] for video_code in self.video_codes], axis=0)
        return all_images

    def analyse_video(self, video_code):
        pass


class FishStimulusMap(StimulusMap):

    def __init__(self, video_directory, tracking_directory, fish_bouts, background):
        StimulusMap.__init__(self, video_directory, tracking_directory, fish_bouts, background)

    def analyse_video(self, video_code):
        video_images = []
        timestamp = video_code_to_timestamp(video_code)
        video_bouts = self.fish_bouts.get_group(video_code)
        video_path = os.path.join(self.video_directory, timestamp + '.avi')
        tracking_path = os.path.join(self.tracking_directory, video_code + '.csv')
        if os.path.exists(tracking_path):
            tracking_info = read_csv(tracking_path, **self.dtype)
            cap = cv2.VideoCapture(video_path)
            for idx, bout_info in video_bouts.iterrows():
                bout_images = []
                for frame_number in (bout_info.start, bout_info.end):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bg = background_division(frame, self.background)
                    ret, threshed = cv2.threshold(bg, 8, 255, cv2.THRESH_BINARY)

                    left, right, heading = tracking_info.loc[frame_number, self.cols].values
                    midpoint = np.array([left, right]).mean(axis=0) / 2.

                    img = rotate_and_centre_image(threshed, midpoint, heading, 0)
                    bout_images.append(img)
                video_images.append(bout_images)
        self.images[video_code] = video_images


class RandomStimulusMap(StimulusMap):

    def __init__(self, video_directory, tracking_directory, fish_bouts, background):
        StimulusMap.__init__(self, video_directory, tracking_directory, fish_bouts, background)

    def analyse_video(self, video_code):
        video_images = []
        timestamp = video_code_to_timestamp(video_code)
        video_bouts = self.fish_bouts.get_group(video_code)
        video_path = os.path.join(self.video_directory, timestamp + '.avi')
        tracking_path = os.path.join(self.tracking_directory, video_code + '.csv')
        if os.path.exists(tracking_path):
            tracking_info = read_csv(tracking_path, **self.dtype)
            cap = cv2.VideoCapture(video_path)
            tracked_frames = tracking_info[tracking_info['tracked']].index
            bout_starts, bout_ends = video_bouts[['start', 'end']].values.T
            exclude_frames = np.concatenate([np.arange(start, end) for (start, end) in zip(bout_starts, bout_ends)])
            frame_numbers = tracked_frames[np.isin(tracked_frames, exclude_frames, assume_unique=True, invert=True)]
            random = np.random.choice(frame_numbers, 100, replace=False)
            for f in random:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bg = background_division(frame, self.background)
                ret, threshed = cv2.threshold(bg, 8, 255, cv2.THRESH_BINARY)

                midpoint, heading = tracking_info.loc[f, self.cols].values
                midpoint = np.array(midpoint) / 2.

                img = rotate_and_centre_image(threshed, midpoint, heading, 0)
                video_images.append(img)
            assert len(video_images) == 100
            video_images = np.array(video_images, dtype='float32')
            average = np.mean(video_images, axis=0)
            variance = np.var(video_images, axis=0)
        else:  # in case there is an error with the tracking
            average, variance = np.empty((2, 500, 504), dtype='float32') * np.nan
        self.images[video_code] = [(average, variance)]


def generate_stimulus_maps(ID):
    # Get fish data
    fish_bouts = mapped_bouts.groupby('ID').get_group(ID)
    fish_info = experiment.data[experiment.data['ID'] == ID].iloc[0]
    video_directory = os.path.join(experiment.video_directory, fish_info.loc['video_directory'])
    tracking_directory = os.path.join(experiment.directory, fish_info.loc['tracking_directory'])
    background = cv2.imread(os.path.join(experiment.directory, fish_info.background_path), 0)
    # Create mapping object
    fish_mapping_multithread = FishStimulusMap(video_directory, tracking_directory, fish_bouts, background)
    images = fish_mapping_multithread.run()
    # Save
    output_path = os.path.join(paramecium_directory, ID + '.npy')
    np.save(output_path, images)


experiment = TrackingExperiment2D('G:\\DATA\\lens_dissection', conditions=True, log=True)
experiment.open()
experiment.subdirs['analysis'] = create_folder(experiment.directory, 'analysis')
paramecium_directory = create_folder(experiment.subdirs['analysis'], 'paramecium_positions')

mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                           index_col=0, dtype={'ID': str, 'video_code': str})
fish_IDs = experiment.data['ID'].values


if __name__ == "__main__":
    _ = Parallel(n_jobs=-1)(delayed(generate_stimulus_maps)(ID) for ID in fish_IDs)
