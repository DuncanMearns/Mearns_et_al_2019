from ..multi_threading import MultiThreading
from ..video import video_code_to_timestamp
from ..tracking import background_division, crop_to_rectangle, rotate_and_centre_image
from ..miscellaneous import read_csv
import cv2
import os
import numpy as np
from ast import literal_eval


class StimulusMap3D(MultiThreading):

    def __init__(self, video_directory, tracking_directory, background_directory, ROI, fish_events):
        MultiThreading.__init__(self, 20)
        self.video_directory = video_directory
        self.tracking_directory = tracking_directory
        self.background_directory = background_directory
        self.ROI = ROI
        self.fish_events = fish_events.groupby('video_code')
        self.video_codes = fish_events['video_code'].unique()
        self.images = {}

    def _run_on_thread(self, arg):
        self.analyse_video(arg)

    def run(self):
        self._run(*self.video_codes)
        all_images = np.concatenate([self.images[video_code] for video_code in self.video_codes], axis=0)
        return all_images

    def analyse_video(self, video_code):
        pass


class FishStimulusMap3D(StimulusMap3D):

    dtype = dict(head_midpoint=literal_eval)
    cols = ['heading', 'head_midpoint', 'fish_elevation', 'side_tracked']

    def __init__(self, video_directory, tracking_directory, background_directory, ROI, fish_events):
        StimulusMap3D.__init__(self, video_directory, tracking_directory, background_directory, ROI, fish_events)

    def analyse_video(self, video_code):
        video_images = []
        timestamp = video_code_to_timestamp(video_code)
        video_events = self.fish_events.get_group(video_code)
        video_path = os.path.join(self.video_directory, timestamp + '.avi')
        tracking_path = os.path.join(self.tracking_directory, video_code + '.csv')
        background_path = os.path.join(self.background_directory, video_code + '.tiff')
        if os.path.exists(tracking_path):
            tracking_info = read_csv(tracking_path, **self.dtype)
            background = cv2.imread(background_path, 0)
            cap = cv2.VideoCapture(video_path)
            for idx, event_info in video_events.iterrows():
                frame_number = event_info.bout_start
                heading, midpoint, elevation, side_tracked = tracking_info.loc[frame_number, self.cols].values
                if side_tracked:  # If side tracking exists
                    cap.set(cv2.CAP_PROP_POS_FRAMES, event_info.start)
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bg = background_division(frame, background)
                    cropped = crop_to_rectangle(bg, *self.ROI)
                    ret, threshed = cv2.threshold(cropped, 8, 255, cv2.THRESH_BINARY)
                    midpoint = np.array(midpoint) / 2.
                    if np.cos(heading) < 0:
                        threshed = threshed[:, ::-1]
                        midpoint[0] = threshed.shape[1] - midpoint[0]
                        elevation *= -1
                    else:
                        elevation = elevation - np.pi
                    img = rotate_and_centre_image(threshed, midpoint, elevation, 0)
                    centre = np.array(img.shape)[::-1] / 2
                    img = crop_to_rectangle(img, centre + (-150, 50), centre + (100, -50))
                else:  # Create empty image
                    img = np.zeros((101, 251), dtype='uint8')
                video_images.append(img)
        self.images[video_code] = video_images


class RandomStimulusMap3D(FishStimulusMap3D):

    def __init__(self, video_directory, tracking_directory, background_directory, ROI, fish_events):
        FishStimulusMap3D.__init__(self, video_directory, tracking_directory, background_directory, ROI, fish_events)

    def analyse_video(self, video_code):
        video_images = []
        timestamp = video_code_to_timestamp(video_code)
        video_path = os.path.join(self.video_directory, timestamp + '.avi')
        tracking_path = os.path.join(self.tracking_directory, video_code + '.csv')
        background_path = os.path.join(self.background_directory, video_code + '.tiff')
        if os.path.exists(tracking_path):
            tracking_info = read_csv(tracking_path, **self.dtype)
            background = cv2.imread(background_path, 0)
            cap = cv2.VideoCapture(video_path)
            tracked_frames = tracking_info[tracking_info['side_tracked']].index
            random = np.random.choice(tracked_frames, 100, replace=False)
            for f in random:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bg = background_division(frame, background)
                cropped = crop_to_rectangle(bg, *self.ROI)
                ret, threshed = cv2.threshold(cropped, 8, 255, cv2.THRESH_BINARY)
                heading, midpoint, elevation, side_tracked = tracking_info.loc[f, self.cols].values
                midpoint = np.array(midpoint) / 2.
                if np.cos(heading) < 0:
                    threshed = threshed[:, ::-1]
                    midpoint[0] = threshed.shape[1] - midpoint[0]
                    elevation *= -1
                else:
                    elevation = elevation - np.pi
                img = rotate_and_centre_image(threshed, midpoint, elevation, 0)
                centre = np.array(img.shape)[::-1] / 2
                img = crop_to_rectangle(img, centre + (-150, 50), centre + (100, -50))
                video_images.append(img)
            assert len(video_images) == 100
            video_images = np.array(video_images, dtype='float32')
            average = np.mean(video_images, axis=0)
            variance = np.var(video_images, axis=0)
        else:  # in case there is an error with the tracking
            average, variance = np.empty((2, 500, 504), dtype='float32') * np.nan
        self.images[video_code] = [(average, variance)]


def generate_stimulus_maps_side(ID, experiment, events_df, output_directory):
    # Get fish data
    fish_events = events_df.groupby('ID').get_group(ID)
    fish_info = experiment.data[experiment.data['ID'] == ID].iloc[0]
    video_directory = os.path.join(experiment.video_directory, fish_info.loc['video_directory'])
    tracking_directory = os.path.join(experiment.directory, fish_info.loc['tracking_directory'])
    background_directory = os.path.join(experiment.directory, fish_info.background_directory)
    side_ROI = fish_info.side_ROI
    # Create mapping object
    fish_mapping_multithread = FishStimulusMap3D(video_directory,
                                                 tracking_directory,
                                                 background_directory,
                                                 side_ROI,
                                                 fish_events)
    images = fish_mapping_multithread.run()
    # Save
    output_path = os.path.join(output_directory, ID + '.npy')
    np.save(output_path, images)


def generate_random_maps_side(ID, experiment, events_df):
    # Get fish data
    fish_info = experiment.data[experiment.data['ID'] == ID].iloc[0]
    video_directory = os.path.join(experiment.video_directory, fish_info.loc['video_directory'])
    tracking_directory = os.path.join(experiment.directory, fish_info.loc['tracking_directory'])
    background_directory = os.path.join(experiment.directory, fish_info.background_directory)
    side_ROI = fish_info.side_ROI
    fish_events = events_df.groupby('ID').get_group(ID)
    # Create mapping object
    random_mapping_multithread = RandomStimulusMap3D(video_directory,
                                                     tracking_directory,
                                                     background_directory,
                                                     side_ROI,
                                                     fish_events)
    video_averages_variances = random_mapping_multithread.run()
    average, variance = np.nanmean(video_averages_variances, axis=0)
    return average, variance
