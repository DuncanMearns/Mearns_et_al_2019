from .image_stimulus_map import image_stimulus_map
from ...multi_threading import MultiThreading
from ...video import Video, video_code_to_timestamp
from ...tracking import background_division
from ...miscellaneous import read_csv
from ...manage_files import get_files
import cv2
import numpy as np
from ast import literal_eval
import os


class StimulusMapper(MultiThreading):

    def __init__(self, experiment, output_directory, **kwargs):
        MultiThreading.__init__(self, **kwargs)
        self.experiment = experiment
        self.IDs = experiment.data['ID'].values
        self.output_directory = output_directory

    def _run_on_thread(self, arg):
        self.analyse_fish(arg)

    def run(self):
        self._run(*self.IDs)

    def analyse_fish(self, ID):
        """Implemented in subclass"""
        return


class BoutStimulusMapper(StimulusMapper):
    """Multi-threaded calculation of stimulus maps for the beginning and end of each bout in an experiment."""

    def __init__(self, bouts, experiment, output_directory, **kwargs):
        StimulusMapper.__init__(self, experiment, output_directory, **kwargs)
        self.bouts = bouts

    def analyse_fish(self, ID):
        fish_bouts = self.bouts.groupby('ID').get_group(ID)
        fish_info = self.experiment.data[self.experiment.data['ID'] == ID].iloc[0]
        output_path = os.path.join(self.output_directory, ID + '.npy')
        tracking_directory = os.path.join(self.experiment.directory, fish_info.tracking_directory)
        background_path = os.path.join(self.experiment.directory, fish_info.background_path)
        fish_maps = []
        for video_code, video_bouts in fish_bouts.groupby('video_code'):
            video_file = video_code_to_timestamp(video_code)
            video_path = os.path.join(self.experiment.video_directory, fish_info.video_directory, video_file + '.avi')
            tracking_path = os.path.join(tracking_directory, video_code + '.csv')
            mapper = VideoStimulusMapper(video_path, background_path, tracking_path)
            maps = mapper.calculate_bout_maps(video_bouts)
            fish_maps.append(maps)
        fish_maps = np.concatenate(fish_maps, axis=0)
        np.save(output_path, fish_maps)


class RandomStimulusMapper(StimulusMapper):

    def __init__(self, n_frames, experiment, output_directory, **kwargs):
        StimulusMapper.__init__(self, experiment, output_directory, **kwargs)
        self.n_frames = n_frames
        self.output_maps = {}

    def run(self):
        super(RandomStimulusMapper, self).run()
        all_maps = np.concatenate([maps for key, maps in self.output_maps.iteritems()], axis=0)
        np.save(os.path.join(self.output_directory, 'random_frames.npy'), all_maps)

    def analyse_fish(self, ID):
        fish_info = self.experiment.data[self.experiment.data['ID'] == ID].iloc[0]
        tracking_directory = os.path.join(self.experiment.directory, fish_info.tracking_directory)
        background_path = os.path.join(self.experiment.directory, fish_info.background_path)
        tracking_files, tracking_paths = get_files(tracking_directory, return_paths=True)
        tracking_pairs = [(f, path) for (f, path) in zip(tracking_files, tracking_paths) if f.endswith('.csv')]
        fish_maps = []
        for tracking_file, tracking_path in tracking_pairs:
            video_code, ext = os.path.splitext(tracking_file)
            video_file = video_code_to_timestamp(video_code)
            video_path = os.path.join(self.experiment.video_directory, fish_info.video_directory, video_file + '.avi')
            mapper = VideoStimulusMapper(video_path, background_path, tracking_path)
            maps = mapper.calculate_random_maps(self.n_frames)
            fish_maps.append(maps)
        fish_maps = np.concatenate(fish_maps, axis=0)
        self.output_maps[ID] = fish_maps


class VideoStimulusMapper():

    def __init__(self, video_path, background_path, tracking_path):
        self.video = Video(video_path)
        self.background = cv2.imread(background_path, 0)
        try:
            self.columns = ['midpoint', 'heading']
            self.tracking = read_csv(tracking_path, midpoint=literal_eval, heading=float)
        except KeyError:
            self.columns = ['left_centre', 'right_centre', 'heading']
            self.tracking = read_csv(tracking_path, **dict(zip(self.columns, [self.convert_type] * 2 + [float])))
            midpoint = np.mean([self.tracking['left_centre'], self.tracking['right_centre']], axis=0)
            self.tracking['midpoint'] = tuple(midpoint)
            self.columns = ['midpoint', 'heading']

    def generate_stimulus_map(self, f):
        self.video.frame_change(f)
        tracking_info = self.tracking.loc[f, self.columns].to_dict()
        img = self.video.grab_frame()[..., 0]
        subtracted = background_division(img, self.background)
        stim_map = image_stimulus_map(subtracted, tracking_info)
        return stim_map

    def calculate_bout_maps(self, bouts):
        video_maps = np.zeros((len(bouts), 2, self.background.shape[0] / 2, self.background.shape[1] / 2),
                              dtype='uint8')
        i = 0
        for idx, bout_info in bouts.iterrows():
            for j, f in enumerate((bout_info.start, bout_info.end)):
                stim_map = self.generate_stimulus_map(f)
                video_maps[i, j] = stim_map
            i += 1
        return video_maps

    def calculate_random_maps(self, n_frames):
        video_maps = np.zeros((n_frames, self.background.shape[0] / 2, self.background.shape[1] / 2), dtype='uint8')
        random_frames = np.random.choice(self.tracking[self.tracking['tracked']].index, n_frames)
        for i, f in enumerate(random_frames):
            stim_map = self.generate_stimulus_map(f)
            video_maps[i] = stim_map
        return video_maps

    @staticmethod
    def convert_type(a):
        return np.array(literal_eval(a))
