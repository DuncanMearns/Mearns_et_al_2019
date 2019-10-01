from ..manage_files import *
from ..miscellaneous import *
from ..tracking import SetBoutDetectionThreshold
import datetime


def string_to_datetime(YYYY_MM_DD, delimiter='_'):
    """Converts a date string into a datetime object.

    Parameters
    ----------
    YYYY_MM_DD : str
        Date in the format (year, month, day) separated by delimiter
    delimiter : str, default '_'
        Delimiter between the year & month and month & day

    Returns
    -------
    date : datetime.date
        The date as a datetime.date object
    """
    digits = YYYY_MM_DD.split(delimiter)
    year, month, day = [int(dig) for dig in digits]
    date = datetime.date(year, month, day)
    return date


def generate_fish_ID(YYYY_MM_DD, idx):
    """Creates a unique ID from the date and fish name.

    Parameters
    ----------
    YYYY_MM_DD : str
        Date in the format YYYY_MM_DD (e.g. '2016_01_30')
    idx : str or int
        Fish name (e.g. 'fish3', 'fish12') or index of the fish in the experiment

    Returns
    -------
    ID : str
        Fish ID in the format 'YYYYMMDDNN' (year, month, day, zero-padded fish number)

    Raises
    ------
    TypeError if idx anything other than a string or int.
    """
    date_ID = YYYY_MM_DD.replace('_', '')
    if type(idx) == int:
        NN = str(idx)
    elif type(idx) == str:
        NN = filter(str.isdigit, idx)
    else:
        raise TypeError('idx must be integer or string')
    if len(NN) == 1:
        NN = '0' + NN
    ID = date_ID + NN
    return ID


def generate_video_code(fish_ID, video_file):
    """Creates a unique ID from the fish ID and a video file

    Parameters
    ----------
    fish_ID : str
        Unique fish identifier
    video_file : str
        A timestamped video file in the format hour-minute-second separated by '-' (e.g.'12-34-56.avi')

    Returns
    -------
    video_code : str
        Video code in the format 'IDHHMMSS' (fish_ID, hour, minute, second)
    """
    timestamp, ext = os.path.splitext(video_file)
    timestamp = timestamp.replace('-', '')
    video_code = fish_ID + timestamp
    return video_code


def custom_ID_func(YYYY_MM_DD, fish_name):
    date_ID = YYYY_MM_DD.replace('_', '')
    fish = fish_name[0].upper() + fish_name[-1]
    ID = date_ID + fish
    return ID


class TrackingExperiment(object):
    """Base class for behaviour tracking experiments.

    Class Attributes
    ----------------
    folders : tuple
        Folders within the experiment directory that contain the output of the tracking.

    Instance Attributes
    -------------------
    directory : str
        The experiment directory where the main data file, backgrounds, tracking files and kinematics files are stored.

    data_name : str, default = 'fish_data'
        The basename of the file that contains information about each fish/trial.

    video_directory : str or None, default = None
        The main video directory where video files for the experiment are stored. By default, assumes that videos are
        contained within a folder named 'videos' in the main experiment directory. Methods that require videos cannot be
        called if a video directory does not exist.

    conditions : bool, default = False
        Whether to include a "condition" column in the data file.

    log : bool, default = True
        Whether to record analysis in the experiment log.

    data_path : str
        The complete path to the main data file. Used for reading and writing. By default the main experiment file is
        called 'fish_data', however this can be set as a keyword argument when initialising the TrackingExperiment2D
        object.

    data : pd.DataFrame
        The main data file that contains information about all the fish in the experiment. Loaded from the data_path.

    subdirs : dict
        Key, value pairs where the key is the name of the folder and the value is the complete path to the folder within
        the experiment directory.

    log_path : str
        Complete path to the analysis log file.

    data_columns : list
        List of strings containing the column names of the data file.

    data_types : dict
        Key, value pairs where the key is one of the data_columns and the value is a function to apply to the elements
        of that column when reading the data file.
    """
    folders = ('backgrounds', 'tracking', 'kinematics', 'analysis')

    def __init__(self, directory, data_name='fish_data', video_directory=None, conditions=False, log=True):

        # set the experiment directory
        if directory is None or not os.path.exists(directory):
            self.directory = pickdir()
        else:
            self.directory = directory

        # set the video directory
        if video_directory is None:
            default_directory = os.path.join(self.directory, 'videos')
            if os.path.exists(default_directory):
                self.video_directory = default_directory
            else:
                print 'Video directory not specified!'
                self.video_directory = None
        else:
            if os.path.exists(video_directory):
                self.video_directory = video_directory
            else:
                print 'Video directory does not exist!'
                self.video_directory = None

        # set the path to the main data file for the experiment
        self.data_name = data_name
        self.data_path = create_filepath(self.directory, self.data_name, '.csv', return_exists=False)

        # create experiment subdirectories
        self.subdirs = {}
        for folder_name in self.folders:
            self.subdirs[folder_name] = create_folder(self.directory, folder_name)

        # create experiment log
        self.log_path = create_filepath(self.directory, 'log', '.txt', return_exists=False)
        self.log = log
        self._update_log(None)

        self.conditions = conditions
        if self.conditions:
            self.data_columns = ['ID', 'date', 'name', 'condition',
                                 'video_directory', 'background_path', 'tracking_directory', 'kinematics_directory']
        else:
            self.data_columns = ['ID', 'date', 'name',
                                 'video_directory', 'background_path', 'tracking_directory', 'kinematics_directory']

        self.data_types = {'ID': str}

    def _read_experiment(self):
        """Opens the main data file and correctly formats columns."""
        if os.path.exists(self.data_path):
            self.data = read_csv(self.data_path, **self.data_types)
        else:
            self.data = pd.DataFrame(columns=self.data_columns)
        if (not self.conditions) and ('condition' in self.data.columns) and (not np.all(pd.isnull(self.data['condition']))):
            self.conditions = True
            self.data_columns.insert(3, 'condition')
            self.data = self.data[self.data_columns]
        elif self.conditions and ('condition' not in self.data.columns):
            self.data['condition'] = np.nan
            self.data = self.data[self.data_columns]

    def _write_experiment(self):
        """Saves the experiment data to the data_path."""
        self.data.loc[:, self.data_columns].to_csv(self.data_path, index=False)

    def _update_log(self, message):
        """Updates the analysis log with a timestamp and message."""
        if self.log:
            timestamp = datetime.datetime.now()
            if not os.path.exists(self.log_path):
                init_message = '-'*26 + '\n{}\n'.format(timestamp) + '-'*26 + '\nExperiment created.'
                with open(self.log_path, 'w') as log:
                    log.write(init_message)
            if message is not None:
                with open(self.log_path, 'a') as log:
                    log.write('\n\n' + '-'*26 + '\n{}\n'.format(timestamp) + '-'*26 + '\n' + message)
            else:
                return

    def _check_video_directory(self):
        """Checks whether the main video directory is specified and exists. If the main video directory is not specified
        the user will be given the option to select a directory in a GUI."""
        if self.video_directory is None:
            select_dir = yes_no_question('Video directory is not specified. Select a directory?')
            if select_dir:
                self.video_directory = pickdir()
            else:
                sys.exit('No video directory specified. Exiting!')

    def _missing_data(self, col):
        """Returns a DataFrame which contains NaN data in the specified column.

        Parameters
        ----------
        col : str
            Name of the column to select NaN data

        Returns
        -------
        missing_data : pd.DataFrame
            DataFrame containing NaN data in the specified col
        """
        return self.data[self.data[col].isnull()]

    def open(self):
        """Opens the experiment."""
        self._read_experiment()
        return self

    def save(self):
        """Saves changes to the experiment."""
        self._write_experiment()
        return self

    def update_entries(self, ID_func=generate_fish_ID):
        """Add new fish to the experiment.

        Parameters
        ----------
        ID_func : function, optional
            The function to call when generating IDs. The default function is 'generate_fish_ID'. User-defined functions
            should take folder names as the first two inputs and return a string.
        """
        self._check_video_directory()
        new_entries = []
        for date_folder, date_directory in zip(*get_directories(self.video_directory)):
            date = string_to_datetime(date_folder)
            for fish_name in get_directories(date_directory, return_paths=False):
                fish_ID = ID_func(date_folder, fish_name)
                if fish_ID not in self.data['ID'].values:
                    fish_video_directory = os.path.join(date_folder, fish_name)
                    fish_entry = dict(ID=fish_ID, date=date, name=fish_name, video_directory=fish_video_directory)
                    if self.conditions:
                        fish_condition = raw_input('Set condition for {}, {}: '.format(date, fish_name))
                        fish_entry['condition'] = fish_condition
                    self.data = self.data.append(fish_entry, ignore_index=True)
                    new_entries.append(fish_ID)
        self._write_experiment()
        if len(new_entries) > 0:
            self._update_log('Added following fish to experiment:\n' + '\n'.join(new_entries))
        return self.data

    def set_conditions(self):
        """Allows user to input the condition for each fish in the experiment."""
        if not self.conditions:
            self.conditions = True
            self.data_columns.insert(3, 'condition')
            self.data['condition'] = np.nan
            self.data = self.data[self.data_columns]
        for idx, fish in self._missing_data('condition').iterrows():
            fish_condition = raw_input('Set condition for {}, {}: '.format(fish.date, fish['name']))
            self.data.loc[idx, 'condition'] = fish_condition
        self._write_experiment()
        return self.data

    def set_bout_detection_thresholds(self, fish_ID, video_ID=None, frame_rate=500.,
                                      default_threshold=0.02, default_min_length=0.05):
        """Set the thresholds for detecting bouts using a GUI.

        Parameters
        ----------
        fish_ID : str or int
            The ID of the fish to use for setting thresholds or the index of the fish within the data file.

        video_ID : str or int, default = None
            The ID of the video to use for setting thresholds or the index of the video within the video folder.

        frame_rate : float, default = 500.
            The frame rate at which the data were acquired.

        default_threshold : float, default = 0.02
            The initial default threshold to use for defining bouts.

        default_min_length : float, default = 0.05
            The initial default minimum time window (in seconds) over which frames above threshold are considered a bout.

        Returns
        -------
        threshold, min_length : float, float
            The threshold and minimum bout length used for finding bouts

        Raises
        ------
        TypeError if fish_ID or video_ID anything other than an int or a str.

        See Also
        --------
        SetBoutDetectionThreshold
        """
        threshold = default_threshold
        min_length = default_min_length
        if type(fish_ID) == str:
            fish = self.data.groupby('ID').get_group(fish_ID).iloc[0]
        elif type(fish_ID) == int:
            fish = self.data.iloc[fish_ID]
        else:
            raise TypeError('fish_ID must be integer or string')
        assert ~pd.isnull(fish.kinematics_directory), 'Kinematics directory not specified!'
        kinematics_directory = os.path.join(self.directory, fish.kinematics_directory)
        kinematic_files, kinematic_paths = get_files(kinematics_directory)
        if video_ID is None:
            for path in kinematic_paths:
                kinematics = pd.read_csv(path)
                tail_tip = kinematics['tip']
                threshold, min_length = SetBoutDetectionThreshold(tail_tip, frame_rate, threshold, min_length).set()
                next_file = yes_no_question('Check next file?')
                if not next_file:
                    break
        else:
            if type(video_ID) == str:
                video_codes = [os.path.splitext(f)[0] for f in kinematic_files]
                video_index = video_codes.index(video_ID)
            elif type(video_ID) == int:
                video_index = video_ID
            else:
                raise TypeError('video_ID must be integer or string')
            path = kinematic_paths[video_index]
            kinematics = pd.read_csv(path)
            tail_tip = kinematics['tip']
            threshold, min_length = SetBoutDetectionThreshold(tail_tip, frame_rate, threshold, min_length).set()
        return threshold, min_length
