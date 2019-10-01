from .tracking_experiment_helpers import *
from ..manage_files import *
from ..miscellaneous import *
from ..tracking import *
from ..analysis.bouts import BoutData, import_bouts
from behaviour_analysis.analysis.alignment.distance import calculate_distance_matrix_templates

import cv2
from ast import literal_eval
from joblib import Parallel, delayed


class TrackingExperiment2D(TrackingExperiment):
    """Class for handling behaviour tracking experiments with a single imaging view in the horizontal plane.

    Parameters
    ----------
    directory : str
        The path to the directory containing the experiment.

    data_name : str, optional
        The basename of the csv data file containing metadata for the experiment (default = 'fish_data').

    video_directory : str, optional
        The main video directory where video files for the experiment are stored. None by default, which assumes that
        videos are contained within a folder named 'videos' in the main experiment directory.

    conditions : bool, optional
        Whether to include a "condition" column in the data file, default = False.

    log : bool, optional
        Whether to record analysis in the experiment log, default = True.
    """

    def __init__(self, directory, data_name='fish_data', video_directory=None, conditions=False, log=True):

        TrackingExperiment.__init__(self, directory, data_name, video_directory, conditions, log)

        self.data_columns += ['thresh1', 'thresh2', 'ROI']

        self.data_types['thresh1'] = int
        self.data_types['thresh2'] = int
        self.data_types['ROI'] = literal_eval

    def calculate_backgrounds(self, sample_factor=100, projection='median', parallel_processing=True, n_processors=-1):
        """Calculate an average projection for all fish in the experiment.

        Parameters
        ----------
        sample_factor : int
            The factor by which to down-sample videos when calculating a projection.

        projection : {'median', 'mean', 'average', 'min', 'minimum', 'max', 'maximum'}
            The kind of projection to calculate to generate a background image.
                'median': calculate a median intensity projection
                'mean' or 'average': calculate a mean intensity projection
                'min' or 'minimum': calculate a minimum intensity projection
                'max' or 'maximum': calculate a maximum intensity projection

        parallel_processing : bool, optional
            Whether or not to distribute analysis over multiple cores (default = True). Each fish is analysed on a
            separate core.

        n_processors : bool, optional
            The number of cores to use when calculating backgrounds if 'parallel_processing' is True. Default = -1 which
            uses all available cores.
        """
        self._check_video_directory()
        timer = Timer()
        fish_idxs = []
        background_paths = []
        all_video_paths = []
        for idx, fish in self._missing_data('background_path').iterrows():
            video_directory = os.path.join(self.video_directory, fish.video_directory)
            assert os.path.exists(video_directory), 'Video directory does not exist!'
            background_file = fish.ID + '.tiff'
            background_path = os.path.join(self.subdirs['backgrounds'], background_file)
            if not os.path.exists(background_path):
                video_files, video_paths = get_files(video_directory)
                fish_idxs.append(idx)
                background_paths.append(background_path)
                all_video_paths.append(video_paths)
            else:
                self.data.loc[idx, 'background_path'] = os.path.join('backgrounds', background_file)
                self._write_experiment()
        if len(fish_idxs) > 0:
            timer.start()
            if parallel_processing:
                # start_time = time.time()
                print 'Calculating backgrounds...',
                fish_times = Parallel(n_jobs=n_processors)(
                    delayed(save_background)(background_path, *video_paths, sample_factor=sample_factor,
                                             projection_type=projection) for background_path, video_paths in
                    zip(background_paths, all_video_paths))
                time_taken = timer.stop()
                print 'done!'
                print 'Total time: {}'.format(timer.convert_time(time_taken))
                print 'Average time per fish: {}'.format(timer.convert_time(np.mean(fish_times)))
                for idx, background_path in zip(fish_idxs, background_paths):
                    self.data.loc[idx, 'background_path'] = os.path.join('backgrounds', os.path.basename(background_path))
                self._write_experiment()
            else:
                for idx, background_path, video_paths in zip(fish_idxs, background_paths, all_video_paths):
                    # start_time = time.time()
                    print 'Calculating background for {}...'.format(self.data.loc[idx, 'ID']),
                    background = calculate_projection(*video_paths, sample_factor=sample_factor, projection_type=projection)
                    background = background.astype('uint8')
                    cv2.imwrite(background_path, background)
                    time_taken = timer.lap()
                    print 'done! ({})'.format(timer.convert_time(time_taken))
                    self.data.loc[idx, 'background_path'] = os.path.join('backgrounds', os.path.basename(background_path))
                    self._write_experiment()
            self._update_log('Background calculation\n'
                             'sample_factor: {}\n'
                             'projection: {}\n'
                             'run time: {}'.format(sample_factor, projection, timer.convert_time(timer.time)))

    def set_thresholds(self, n_points, track_eyes=True):
        """Set the thresholds for finding the fish and internal contours for videos.

        Parameters
        ----------
        n_points : int
            The number of points to fit to the tail.

        track_eyes : bool, optional
            Whether to perform eye tracking (default = True)
        """
        self._check_video_directory()
        for idx, fish in self._missing_data('thresh1').iterrows():
            video_directory = os.path.join(self.video_directory, fish.video_directory)
            assert os.path.exists(video_directory), 'Video directory does not exist!'
            assert ~pd.isnull(fish.background_path), 'Background path not specified!'
            background_path = os.path.join(self.directory, fish.background_path)
            assert os.path.exists(background_path), 'Background path does not exist!'
            video_files, video_paths = get_files(video_directory)
            thresh1, thresh2 = set_thresholds(video_paths, background_path, n_points=n_points, track_eyes=track_eyes)
            self.data.loc[idx, 'thresh1'] = thresh1
            self.data.loc[idx, 'thresh2'] = thresh2
            self._write_experiment()

    def run_tracking(self, n_points, track_eyes=True, parallel_processing=True, n_processors=-1):
        """Performs tracking for all un-tracked videos in an experiment and saves a csv containing positional and
        rotational information and a npy file containing tail points across all frames.

        Parameters
        ----------
        n_points : int
            The number of points to fit to the tail.

        track_eyes : bool, optional
            Whether to perform eye tracking (default = True).

        parallel_processing : bool, optional
            Whether videos should be processed in parallel on multiple cores (default = True). Setting this parameter to
            False will slow down the tracking but may be useful for de-bugging.

        n_processors : int, optional
            Number of cores to use (if 'parallel_processing' = True). By default, uses all available cores.
        """
        self._check_video_directory()
        if track_eyes:
            tracking_function = track_video
        else:
            tracking_function = track_video_tail_only
        timer = Timer()
        for idx, fish in self.data.iterrows():
            video_directory = os.path.join(self.video_directory, fish.video_directory)
            assert os.path.exists(video_directory), 'Video directory does not exist!'
            assert ~pd.isnull(fish.background_path), 'Background path not specified!'
            background_path = os.path.join(self.directory, fish.background_path)
            assert os.path.exists(background_path), 'Background path does not exist!'
            assert ~pd.isnull(fish.loc[['thresh1', 'thresh2']]).any(), 'Thresholds have not been set!'

            tracking_directory = create_folder(self.subdirs['tracking'], fish.ID)
            if pd.isnull(fish.loc['tracking_directory']):
                self.data.loc[idx, 'tracking_directory'] = os.path.join('tracking', fish.ID)
                self._write_experiment()

            # CREATE FILE PATHS FOR TRACKING
            video_paths = []
            video_codes = []
            for video_file, video_path in zip(*get_files(video_directory)):
                video_code = generate_video_code(fish.ID, video_file)
                tracking_path, path_exists = create_filepath(tracking_directory, video_code, '.csv', return_exists=True)
                if not path_exists:
                    video_paths.append(video_path)
                    video_codes.append(video_code)
            assert len(video_paths) == len(video_codes)

            if len(video_paths) > 0:
                print fish.ID
                timer.start()
                background = cv2.imread(background_path, 0)
                fish_kwargs = {'background': background,
                               'thresh1': fish.thresh1,
                               'thresh2': fish.thresh2,
                               'n_points': n_points,
                               'save_output': True,
                               'output_directory': tracking_directory}
                if parallel_processing:
                    video_times = Parallel(n_jobs=n_processors)(delayed(tracking_function)(video_path=video_path, filename=video_code, **fish_kwargs)
                                                                for (video_path, video_code) in zip(video_paths, video_codes))
                    total_time_taken = timer.stop()
                    average_time_taken = np.mean(video_times)
                    print 'Total time: {}'.format(timer.convert_time(total_time_taken))
                    print 'Average time per video: {}'.format(timer.convert_time(average_time_taken))
                else:
                    for video_path, video_code in zip(video_paths, video_codes):
                        print '\t{}...'.format(os.path.basename(video_path)),
                        video_time = tracking_function(video_path=video_path, filename=video_code, **fish_kwargs)
                        print 'done! {}'.format(timer.convert_time(video_time))
                self._update_log('Fish {} tracked\n'
                                 'n_points: {}'.format(fish.ID, n_points))

    def check_tracking(self, fish_ID, video_ID=None):
        """Checks the tracking for a video (or videos).

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
                k = check_tracking(video_path, csv_path, points_path, winname='Check tracking: {}'.format(video_file))
                if k == KeyboardInteraction().esc or k == KeyboardInteraction().enter:
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
        check_tracking(video_path, csv_path, points_path, winname='Check tracking: {}'.format(video_files[video_index]))
        return

    def calculate_kinematics(self, frame_rate, parallel_processing=True, n_processors=-1):
        """Extracts kinematic data from the tracking files for each video and saves the output in a csv file.

        Parameters
        ----------
        frame_rate : float
            The frame rate at which the data were acquired.

        parallel_processing : bool, optional
            Whether videos should be processed in parallel on multiple cores (default = True). Setting this parameter to
            False will slow down the tracking but may be useful for de-bugging.

        n_processors : int, optional
            Number of cores to use (if 'parallel_processing' = True). By default, uses all available cores.
        """
        timer = Timer()
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
                    timer.start()
                    analysis_times = Parallel(n_jobs=n_processors)(
                        delayed(calculate_kinematics)(csv_path, points_path, frame_rate, save_output=True, output_path=output_path)
                        for (csv_path, points_path, output_path) in zip(csv_paths, points_paths, output_paths))
                    total_time_taken = timer.stop()
                    average_time_taken = np.mean(analysis_times)
                    print 'Total time: {}'.format(timer.convert_time(total_time_taken))
                    print 'Average time per file: {}'.format(timer.convert_time(average_time_taken))
                else:
                    for csv_path, points_path, output_path in zip(csv_paths, points_paths, output_paths):
                        file_code = os.path.splitext(os.path.basename(csv_path))[0]
                        print '\t{}...'.format(file_code),
                        analysis_time = calculate_kinematics(csv_path, points_path, frame_rate, save_output=True, output_path=output_path)
                        print 'done! {}'.format(timer.convert_time(analysis_time))
                self._update_log('Fish {} kinematics extracted\n'
                                 'frame_rate: {}'.format(fish.ID, frame_rate))

    def set_ROIs(self):
        """Specify a region of interest within the videos for each fish."""
        updated_fish = []
        for idx, fish in self._missing_data('ROI').iterrows():
            assert ~pd.isnull(fish.background_path), 'Background path not specified!'
            background_path = os.path.join(self.directory, fish.background_path)
            assert os.path.exists(background_path), 'Background path does not exist!'
            ROI = RegionOfInterest(background_path).select()
            self.data.loc[idx, 'ROI'] = str(ROI)
            updated_fish.append(fish.ID)
            self._write_experiment()
            self._read_experiment()
        if len(updated_fish) > 0:
            self._update_log('Set ROIs for following IDs:\n' + '\n'.join(updated_fish))

    def get_bouts(self, frame_rate=500., threshold=0.02, min_length=0.05, check_ROI=True):
        """Find all the bouts in the experiment and saves the result in a csv file.

        Parameters
        ----------
        frame_rate : float, optional
            The frame rate at which the data were acquired. Default is 500 fps.

        threshold : float, optional
            The threshold to use for defining bouts.

        min_length : float, optional
            The minimum time window (in seconds) over which frames above threshold are considered a bout.

        check_ROI : bool, optional
            Check whether the tail falls within the ROI over the entire duration of each bout. Default is True.

        Returns
        -------
        bouts_df : pd.DataFrame
            DataFrame containing every bout in the experiment. Provides the fish ID, video code, and first and last
            frames of each bout. An optional column states whether the bout falls within the ROI.

        Notes
        --------
        - Values for threshold and min_length can be optimised using the 'set_bout_detection_thresholds' method.
        - This method will simply open and return the bouts DataFrame if all videos have been checked for bouts.
        """
        bout_columns = ['ID', 'video_code', 'start', 'end']
        if check_ROI:
            bout_columns.append('ROI')
        bouts_output_path = create_filepath(self.directory, 'bouts', '.csv')
        if os.path.exists(bouts_output_path):
            bouts_df = read_csv(bouts_output_path, ID=str, video_code=str)
        else:
            bouts_df = pd.DataFrame(columns=bout_columns)
        updated_fish = []
        for idx, fish in self.data.iterrows():
            if ~bouts_df['ID'].isin([fish.ID]).any():
                print fish.ID
                updated_fish.append(fish.ID)
                assert ~pd.isnull(fish.kinematics_directory), 'Kinematics directory not specified!'
                kinematics_directory = os.path.join(self.directory, fish.kinematics_directory)
                assert os.path.exists(kinematics_directory), 'Kinematics directory does not exist!'
                tracking_directory = os.path.join(self.directory, fish.tracking_directory)
                if check_ROI:
                    assert ~pd.isnull(fish.ROI), 'ROI not set!'
                    assert os.path.exists(tracking_directory), 'Tracking directory does not exist!'
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
                            in_ROI = [check_points_in_ROI(fish.ROI, video_points[start:end+1]) for (start, end) in bouts_in_video]
                            fish_bouts['ROI'] += in_ROI
                fish_bouts = pd.DataFrame(fish_bouts, columns=bout_columns)
                bouts_df = pd.concat([bouts_df, fish_bouts], ignore_index=True)
                bouts_df.to_csv(bouts_output_path, index=False)
        if len(updated_fish) > 0:
            self._update_log('Extracted bouts from following IDs:\n' + '\n'.join(updated_fish) + '\n\n' +
                             'frame_rate: {}\n'
                             'threshold: {}\n'
                             'min_length: {}'.format(frame_rate, threshold, min_length))
        return bouts_df


class MappingExperiment(TrackingExperiment2D):

    def __init__(self, directory, parent, **kwargs):
        TrackingExperiment2D.__init__(self, directory, **kwargs)
        if isinstance(parent, TrackingExperiment2D):
            self.parent = parent
        elif isinstance(parent, str):
            self.parent = TrackingExperiment2D(parent, conditions=False, log=False)
        else:
            raise TypeError('mapping_experiment must be a TrackingExperiment2D instance or a valid path.')
        self.parent.open()

    def map_bouts(self, n_dims, frame_rate):

        # Import data from mapping experiment
        mapping_space_directory = os.path.join(self.parent.subdirs['analysis'], 'behaviour_space')
        eigenfish = np.load(os.path.join(mapping_space_directory, 'eigenfish.npy'))
        mean_tail, std_tail = np.load(os.path.join(mapping_space_directory, 'tail_statistics.npy'))
        exemplar_info = pd.read_csv(os.path.join(self.parent.subdirs['analysis'], 'exemplars.csv'),
                                    index_col='bout_index',
                                    dtype={'ID': str, 'video_code': str})
        exemplar_info = exemplar_info[exemplar_info['clean']]
        exemplars = BoutData.from_directory(exemplar_info, self.parent.subdirs['kinematics'],
                                            check_tail_lengths=False, tail_columns_only=True)
        exemplars = exemplars.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)
        exemplars = exemplars.list_bouts(values=True, ndims=n_dims)

        # Set paths
        output_directory = create_folder(self.subdirs['analysis'], 'distance_matrices')

        # Import experiment bouts
        experiment_bouts = import_bouts(self.directory)
        experiment_bouts = experiment_bouts.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)

        # Compute distance matrices
        print_heading('CALCULATING DISTANCE MATRICES')
        distances = {}
        analysis_times = []
        timer = Timer()
        timer.start()
        for ID in experiment_bouts.metadata['ID'].unique():
            output_path, path_exists = create_filepath(output_directory, ID, '.npy', True)
            if path_exists:
                distances[ID] = np.load(output_path)
            if not path_exists:
                print ID + '...',
                queries = experiment_bouts.list_bouts(IDs=[ID], values=True, ndims=n_dims)
                fish_distances = calculate_distance_matrix_templates(queries, exemplars, fs=frame_rate)
                distances[ID] = fish_distances
                time_taken = timer.lap()
                analysis_times.append(time_taken)
                print timer.convert_time(time_taken)
                np.save(output_path, fish_distances)
        if len(analysis_times) > 0:
            print 'Average time: {}'.format(timer.convert_time(timer.average))

        # Assign exemplars
        mapped_bouts = experiment_bouts.metadata.copy()
        mapped_bouts['exemplar'] = None
        for ID, fish_distances in distances.iteritems():
            bout_idxs = mapped_bouts[mapped_bouts['ID'] == ID].index
            nearest_exemplar = np.argmin(fish_distances, axis=1)
            mapped_bouts.loc[bout_idxs, 'exemplar'] = nearest_exemplar
        mapped_bouts.to_csv(os.path.join(self.subdirs['analysis'], 'mapped_bouts.csv'))
