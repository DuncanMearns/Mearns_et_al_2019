from .tracking_experiment_helpers import *
from ..tracking import *
from ..tracking3D import *
from ..miscellaneous import Timer

from joblib import Parallel, delayed
from ast import literal_eval


class TrackingExperiment3D(TrackingExperiment):
    """Class for handling 2D behaviour tracking experiments

    Class Attributes
    ----------------
    data_columns : tuple
        Column names of the main data file for the experiment:
            'ID' : str
                A unique identifier for the fish in the experiment (format: 'YYYYMMDDNN')

            'date' (datetime.date) :
                The date the animal was tested

            'name' : str
                The name of the folder where the video files for the fish are kept

            'condition' : str
                The experimental condition of the fish

            'video_directory' : str
                The subdirectory within the main video directory where videos for the fish are stored

            'background_path' : str
                The path to the background file within the main experiment directory

            'tracking_directory' : str
                The subdirectory within the main experiment directory where tracking files for the fish are stored

            'kinematics_directory' : str
                The subdirectory within the main experiment directory where kinematic files for the fish are stored

            'thresh1' : int (0-255)
                The threshold used to find the outline of the fish in video files

            'thresh2' : int (0-255)
                The threshold used to the the outline of the eyes and swim bladder in video files

    Instance Attributes
    -------------------
    directory : str
        The main experiment directory where the main data file, backgrounds, tracking files and kinematics files are stored.

    video_directory : str or None
        The main video directory where video files for the experiment are stored. By default, assumes that videos are
        contained within a folder named 'videos' in the main experiment directory. Methods that require videos cannot be
        called if a video directory does not exist.

    data_path : str
        The complete path to the main data file. Used for reading and writing. By default the main experiment file is
        called 'fish_data', however this can be set as a keyword argument when initialising the TrackingExperiment2D
        object.

    data : pd.DataFrame
        The main data file that contains information about all the fish in the experiment.
    """

    def __init__(self, directory, data_name='fish_data', video_directory=None, conditions=False, log=True):

        TrackingExperiment.__init__(self, directory, data_name, video_directory, conditions, log)

        self.data_columns += ['top_threshes', 'side_threshes', 'top_ROI', 'side_ROI']
        self.data_columns[self.data_columns.index('background_path')] = 'background_directory'

        self.data_types['top_threshes'] = literal_eval
        self.data_types['side_threshes'] = literal_eval
        self.data_types['top_ROI'] = literal_eval
        self.data_types['side_ROI'] = literal_eval

    def calculate_backgrounds(self, sample_factor=100, projection='median', parallel_processing=True, n_processors=-1):
        """Calculate an average projection for each video for all fish in the experiment"""
        self._check_video_directory()
        for idx, fish in self._missing_data('background_directory').iterrows():
            video_directory = os.path.join(self.video_directory, fish.video_directory)
            assert os.path.exists(video_directory), 'Video directory does not exist!'

            background_directory = create_folder(self.subdirs['backgrounds'], fish.ID)
            if pd.isnull(fish.loc['background_directory']):
                self.data.loc[idx, 'background_directory'] = os.path.join('backgrounds', fish.ID)
                self._write_experiment()

            video_files, video_paths = get_files(video_directory)
            video_codes = [generate_video_code(fish.ID, video_file) for video_file in video_files]
            background_paths = [create_filepath(background_directory, video_code, '.tiff', return_exists=True) for video_code in video_codes]
            video_background_pairs = [(video_path, background_path) for (video_path, (background_path, path_exists)) in zip(video_paths, background_paths) if not path_exists]

            if len(video_background_pairs) > 0:
                print fish.ID
                background_kwargs = {'projection': projection,
                                     'sample_factor': sample_factor}
                if parallel_processing:
                    fish_start = time.time()
                    video_times = Parallel(n_jobs=n_processors)(delayed(save_background)(bg_path, v_path, **background_kwargs) for (v_path, bg_path) in video_background_pairs)
                    fish_end = time.time()
                    print 'Total time: {} minutes'.format((fish_end - fish_start) / 60.)
                    print 'Average time per video: {} minutes'.format(np.mean(video_times) / 60.)
                else:
                    for video_path, background_path in video_background_pairs:
                        print '\t{}...'.format(os.path.basename(video_path)),
                        video_time = save_background(background_path, video_path, **background_kwargs)
                        print 'done! {} minutes'.format(video_time / 60.)

    def set_ROIs(self):
        for idx, fish in self._missing_data('top_ROI').iterrows():
            assert ~pd.isnull(fish.background_path), 'Background directory not specified!'
            background_directory = os.path.join(self.directory, fish.background_directory)
            assert os.path.exists(background_directory), 'Background directory does not exist!'
            background_files, background_paths = get_files(background_directory)
            side_ROI = RegionOfInterest(background_paths[0], winname='side ROI').select()
            top_ROI = RegionOfInterest(background_paths[0], winname='top ROI').select()
            self.data.loc[idx, 'top_ROI'] = str(top_ROI)
            self.data.loc[idx, 'side_ROI'] = str(side_ROI)
            self._write_experiment()
            self._read_experiment()

    def set_thresholds(self, n_points):
        """Set the thresholds for finding the fish and internal contours for videos"""
        self._check_video_directory()
        for idx, fish in self._missing_data('top_threshes').iterrows():
            video_directory = os.path.join(self.video_directory, fish.video_directory)
            assert os.path.exists(video_directory), 'Video directory does not exist!'
            assert ~pd.isnull(fish.background_directory), 'Background directory not specified!'
            background_directory = os.path.join(self.directory, fish.background_directory)
            assert os.path.exists(background_directory), 'Background directory does not exist!'
            video_files, video_paths = get_files(video_directory)
            background_files, background_paths = get_files(background_directory)
            thresh1, thresh2, thresh3, thresh4 = set_thresholds_3d(video_paths, background_paths, fish.top_ROI,
                                                                   fish.side_ROI, n_points=n_points)
            self.data.loc[idx, 'top_threshes'] = str((thresh1, thresh2))
            self.data.loc[idx, 'side_threshes'] = str((thresh3, thresh4))
            self._write_experiment()

    def run_tracking(self, n_points, parallel_processing=True, n_processors=-1):
        """Performs tracking for all un-tracked videos in an experiment and saves a csv containing positional and
        rotational information and a npy file containing tail points across all frames

        Parameters
        ----------
        n_points : int (default = 51)
            Number of points to fit to the tail

        parallel_processing : bool, default True
            Whether videos should be processed in parallel on multiple processors. Setting this parameter to False will
            slow down the tracking but may be useful for de-bugging.

        n_processors : int (default = -1)
            Number of processors to use (if parallel_processing = True). By default, uses all available processors.
        """
        self._check_video_directory()
        for idx, fish in self.data.iterrows():
            video_directory = os.path.join(self.video_directory, fish.video_directory)
            assert os.path.exists(video_directory), 'Video directory does not exist!'
            assert ~pd.isnull(fish.background_directory), 'Background directory not specified!'
            background_directory = os.path.join(self.directory, fish.background_directory)
            assert os.path.exists(background_directory), 'Background directory does not exist!'
            assert ~pd.isnull(fish.loc[['top_ROI', 'side_ROI']]).any(), 'ROIs have not been set!'
            assert ~pd.isnull(fish.loc[['top_threshes', 'side_threshes']]).any(), 'Thresholds have not been set!'

            tracking_directory = create_folder(self.subdirs['tracking'], fish.ID)
            if pd.isnull(fish.loc['tracking_directory']):
                self.data.loc[idx, 'tracking_directory'] = os.path.join('tracking', fish.ID)
                self._write_experiment()

            # CREATE FILE PATHS FOR TRACKING
            video_paths = []
            video_codes = []
            background_paths = []
            for video_file, video_path in zip(*get_files(video_directory)):
                video_code = generate_video_code(fish.ID, video_file)
                background_path, path_exists = create_filepath(background_directory, video_code, '.tiff', return_exists=True)
                assert path_exists, 'Background path does not exist!'
                tracking_path, path_exists = create_filepath(tracking_directory, video_code, '.csv', return_exists=True)
                if not path_exists:
                    video_paths.append(video_path)
                    video_codes.append(video_code)
                    background_paths.append(background_path)
            assert len(video_paths) == len(video_codes)

            if len(video_paths) > 0:
                print fish.ID
                fish_kwargs = {'top_ROI': fish.top_ROI,
                               'side_ROI': fish.side_ROI,
                               'thresh1': fish.top_threshes[0],
                               'thresh2': fish.top_threshes[1],
                               'thresh3': fish.side_threshes[0],
                               'thresh4': fish.side_threshes[1],
                               'n_points': n_points,
                               'save_output': True,
                               'output_directory': tracking_directory}
                if parallel_processing:
                    timer = Timer()
                    timer.start()
                    video_times = Parallel(n_jobs=n_processors)(delayed(track_video_3d)(video_path=video_path, background_path=background_path, filename=video_code, **fish_kwargs)
                                                                for (video_path, background_path, video_code) in zip(video_paths, background_paths, video_codes))
                    total_time_taken = timer.stop()
                    average_time_taken = np.mean(video_times)
                    print 'Total time: {}'.format(timer.convert_time(total_time_taken))
                    print 'Average time per video: {}'.format(timer.convert_time(average_time_taken))
                else:
                    for video_path, background_path, video_code in zip(video_paths, background_paths, video_codes):
                        print '\t{}...'.format(os.path.basename(video_path)),
                        video_time = track_video_3d(video_path=video_path, background_path=background_path, filename=video_code, **fish_kwargs)
                        print 'done! {}'.format(Timer.convert_time(video_time))

    def check_tracking(self, fish_ID, video_ID=None):
        """Checks the tracking for a video (or videos)

        Parameters
        ----------
        fish_ID : str or int
            The ID of the fish as a string or index in the data DataFrame.

        video_ID : str or int or None (default = None)
            The ID of the video as a string or index. If None, then each video is checked in turn.

        Raises TypeError if fish_ID or video_ID are anything other than str or int
        """
        if type(fish_ID) == str:
            fish = self.data.groupby('ID').get_group(fish_ID).iloc[0]
        elif type(fish_ID) == int:
            fish = self.data.iloc[fish_ID]
        else:
            raise TypeError('fish_ID must be integer or string')
        self._check_video_directory()
        assert ~pd.isnull(fish.loc['tracking_directory']), 'Tracking directory does not exist!'
        video_directory = os.path.join(self.video_directory, fish.video_directory)
        video_files, video_paths = get_files(video_directory, return_paths=True)
        tracking_directory = os.path.join(self.directory, fish.tracking_directory)
        tracking_files, tracking_paths = get_files(tracking_directory, return_paths=True)
        csv_paths, points_paths = tracking_paths[::2], tracking_paths[1::2]
        assert len(csv_paths) == len(points_paths) <= len(video_paths), 'Incorrect number of files in tracking folder!'
        if video_ID is None:
            for video_path, video_file, csv_path, points_path in zip(video_paths, video_files, csv_paths, points_paths):
                v = check_tracking_3d(video_path, csv_path, points_path, fish.top_ROI, fish.side_ROI, winname='Check tracking: {}'.format(video_file))
                if v.esc() or v.enter():
                    break
            return
        elif type(video_ID) == str:
            video_codes = [generate_video_code(fish.ID, f) for f in video_files]
            video_index = video_codes.index(video_ID)
        elif type(video_ID) == int:
            video_index = video_ID
        else:
            raise TypeError('video_ID must be integer or string')
        video_path, csv_path, points_path = video_paths[video_index], csv_paths[video_index], points_paths[video_index]
        check_tracking_3d(video_path, csv_path, points_path, winname='Check tracking: {}'.format(video_files[video_index]))
        return

    def calculate_kinematics(self, frame_rate, parallel_processing=True, n_processors=-1):

        for idx, fish in self.data.iterrows():
            assert ~pd.isnull(fish.tracking_directory), 'Tracking directory not specified!'
            tracking_directory = os.path.join(self.directory, fish.tracking_directory)
            assert os.path.exists(tracking_directory), 'Tracking directory does not exist!'

            # Create output directory
            fish_kinematics_directory = create_folder(self.subdirs['kinematics'], fish.ID)
            if pd.isnull(fish.loc['kinematics_directory']):
                self.data.loc[idx, 'kinematics_directory'] = os.path.join('kinematics', fish.ID)
                self._write_experiment()

            tracking_files, tracking_paths = get_files(tracking_directory)
            csv_files = tracking_files[::2]
            csv_paths, points_paths = tracking_paths[::2], tracking_paths[1::2]
            assert len(csv_paths) == len(points_paths), 'Incorrect number of files in tracking folder!'
            output_paths, paths_exist = zip(*[create_filepath(fish_kinematics_directory, f, '', return_exists=True) for f in csv_files])
            if not np.all(paths_exist):
                print fish.ID
                if parallel_processing:
                    timer = Timer()
                    timer.start()

                    analysis_times = Parallel(n_jobs=n_processors)(
                        delayed(calculate_kinematics_3d)(csv_path, points_path, frame_rate, save_output=True, output_path=output_path)
                        for (csv_path, points_path, output_path) in zip(csv_paths, points_paths, output_paths))

                    total_time_taken = timer.stop()
                    average_time_taken = np.mean(analysis_times)
                    print 'Total time: {}'.format(timer.convert_time(total_time_taken))
                    print 'Average time per video: {}'.format(timer.convert_time(average_time_taken))
                else:
                    for csv_path, points_path, output_path in zip(csv_paths, points_paths, output_paths):
                        file_code = os.path.splitext(os.path.basename(csv_path))[0]
                        print '\t{}...'.format(file_code),
                        analysis_time = calculate_kinematics_3d(csv_path, points_path, frame_rate, save_output=True, output_path=output_path)
                        print 'done! {}'.format(Timer.convert_time(analysis_time))

    def get_bouts(self, frame_rate=400., threshold=0.02, min_length=0.05, check_ROI=False):
        bout_columns = ['ID', 'video_code', 'start', 'end']
        if check_ROI:
            bout_columns.append('ROI')
        bouts_output_path = create_filepath(self.directory, 'bouts', '.csv')
        if os.path.exists(bouts_output_path):
            bouts_df = read_csv(bouts_output_path, ID=str, video_code=str)
        else:
            bouts_df = pd.DataFrame(columns=bout_columns)
        for idx, fish in self.data.iterrows():
            if ~bouts_df['ID'].isin([fish.ID]).any():
                print fish.ID
                assert ~pd.isnull(fish.kinematics_directory), 'Kinematics directory not specified!'
                kinematics_directory = os.path.join(self.directory, fish.kinematics_directory)
                assert os.path.exists(kinematics_directory), 'Kinematics directory does not exist!'
                tracking_directory = os.path.join(self.directory, fish.tracking_directory)
                if check_ROI:
                    assert ~pd.isnull(fish.top_ROI), 'ROI not set!'
                    assert os.path.exists(tracking_directory), 'Tracking directory does not exist!'
                    ROI = np.array(fish.top_ROI)
                    ROI -= ROI[0]
                    ROI[0] += 10
                    ROI[1] -= 10
                kinematics_files, kinematics_paths = get_files(kinematics_directory)
                video_codes = [os.path.splitext(filename)[0] for filename in kinematics_files]
                fish_bouts = dict([(col, []) for col in bout_columns])
                for path, video_code in zip(kinematics_paths, video_codes):
                    bouts_in_video = find_video_bouts(path, frame_rate, threshold, min_length)
                    if len(bouts_in_video) > 0:
                        first_frames, last_frames = zip(*bouts_in_video)
                        fish_bouts['ID'] += [fish.ID] * len(first_frames)
                        fish_bouts['video_code'] += [video_code] * len(first_frames)
                        fish_bouts['start'] += first_frames
                        fish_bouts['end'] += last_frames
                        if check_ROI:
                            points_path = os.path.join(tracking_directory, video_code + '.npy')
                            video_points = np.load(points_path)
                            in_ROI = [check_points_in_ROI(ROI, video_points[start:end+1]) for (start, end) in bouts_in_video]
                            fish_bouts['ROI'] += in_ROI
                fish_bouts = pd.DataFrame(fish_bouts, columns=bout_columns)
                bouts_df = pd.concat([bouts_df, fish_bouts], ignore_index=True)
                bouts_df.to_csv(bouts_output_path, index=False)
        return bouts_df

    def get_jaw_movements(self):
        jaw_columns = ['ID', 'video_code', 'start', 'end']
        jaw_output_path = create_filepath(self.directory, 'jaw_movements', '.csv')
        if os.path.exists(jaw_output_path):
            jaw_df = read_csv(jaw_output_path, ID=str, video_code=str)
        else:
            jaw_df = pd.DataFrame(columns=jaw_columns)
        for idx, fish in self.data.iterrows():
            if ~jaw_df['ID'].isin([fish.ID]).any():
                print fish.ID
                assert ~pd.isnull(fish.kinematics_directory), 'Kinematics directory not specified!'
                kinematics_directory = os.path.join(self.directory, fish.kinematics_directory)
                assert os.path.exists(kinematics_directory), 'Kinematics directory does not exist!'
                kinematics_files, kinematics_paths = get_files(kinematics_directory)
                video_codes = [os.path.splitext(filename)[0] for filename in kinematics_files]
                fish_jaw = dict([(col, []) for col in jaw_columns])
                for path, video_code in zip(kinematics_paths, video_codes):
                    jaw_movements_in_video = find_video_jaw_movements(path)
                    if len(jaw_movements_in_video) > 0:
                        first_frames, last_frames = zip(*jaw_movements_in_video)
                        fish_jaw['ID'] += [fish.ID] * len(first_frames)
                        fish_jaw['video_code'] += [video_code] * len(first_frames)
                        fish_jaw['start'] += first_frames
                        fish_jaw['end'] += last_frames
                fish_jaw = pd.DataFrame(fish_jaw, columns=jaw_columns)
                jaw_df = pd.concat([jaw_df, fish_jaw], ignore_index=True)
                jaw_df.to_csv(jaw_output_path, index=False)
        return jaw_df
