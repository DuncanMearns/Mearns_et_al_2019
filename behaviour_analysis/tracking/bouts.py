from ..miscellaneous import find_contiguous, find_subsequence

import numpy as np
import pandas as pd
from scipy.signal import gaussian
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def find_bouts(k, fs, threshold=0.02, min_length=0.05):
    """Finds bouts using tip angle data

    First, a 100 ms sliding window is applied to the absolute values of the derivative of the tail angles over time.
    Then, frames that are above threshold are found. Finally, only contiguous chunks of time that are above threshold
    and longer than a minimum length are considered as bouts.

    Parameters
    ----------
    k : pd.Series
        The angle of the tip of the tail for all frames in continuously tracked data
    fs : float
        The sampling frequency (frames per second)
    threshold : float, optional (default = 0.02)
        Threshold used for detecting bouts
    min_length : float, optional (default = 0.05)
        The minimum length of time a bout can last (seconds)

    Returns
    -------
    movement_frames : list
        A list of arrays where each array contains the frame numbers for a single bout

    See Also
    --------
    SetBoutDetectionThreshold
    """
    window = int(min_length * fs)
    kernel = gaussian(window * 2, (window * 2) / 5.)
    kernel /= np.sum(kernel)
    diffed = k.diff().shift(-1)
    diffed.iloc[-1] = 0
    abs_diffed = diffed.abs()
    filtered = np.convolve(abs_diffed, kernel, mode='same')
    above_threshold = np.where(filtered > threshold)[0]
    movement_indices = find_contiguous(above_threshold, minsize=int(min_length * fs))
    movement_frames = []
    for frame_indices in movement_indices:
        try:
            movement_frames.append(k.index[frame_indices])
        except IndexError:
            pass
    return movement_frames


def split_long_bout(tip, fs):
    """Splits a long bout, which might occur as a result of two or more separate bouts erroneously being fused together

    Different and slightly more involved method of finding bouts than find_bouts

    Parameters
    ----------
    tip : pd.Series
        The angle of the tip of the tail for the frames that are to be split
    fs : float
        The sampling frequency (frames per second)

    Returns
    -------
    bout_frames : list
        A list of arrays where each array contains the frame numbers for a single bout

    See Also
    --------
    find_bouts
    """
    kernel_width = int(fs * 0.03)
    x = np.arange(kernel_width)
    kernel = np.cos((2 * np.pi * x) / (kernel_width - 1))

    conv = np.convolve(tip, kernel, 'same')
    filtered = pd.Series(np.abs(conv)).rolling(window=15, min_periods=0, center=True).max()
    smoothed = filtered.rolling(window=15, min_periods=0, center=True).mean().values

    frames = np.arange(len(tip))

    masked = np.zeros((len(tip),))
    masked[np.where(smoothed >= 2)] = 1
    masked[np.where(smoothed <= 1)] = -1

    sections = np.split(frames, np.where(np.diff(masked) != 0)[0] + 1)
    landscape = np.array([masked[section[0]] for section in sections])

    bout_lims = []
    peak_idxs = find_subsequence(np.diff(landscape), np.array([1, -1]))
    if len(peak_idxs) > 0:
        peak_idxs[:, 1] += 1
        for peak in peak_idxs:
            if peak[0] == 0:
                start_section = sections[0]
            elif landscape[peak[0] - 1] < landscape[peak[0]]:
                start_section = sections[peak[0] - 1]
            else:
                start_section = sections[peak[0]]
            if peak[1] == len(sections) - 1:
                end_section = sections[-1]
            elif landscape[peak[1] + 1] < landscape[peak[0]]:
                end_section = sections[peak[1] + 1]
            else:
                end_section = sections[peak[1]]

            lims = []
            for i, section in enumerate((start_section, end_section)):
                turning_point_idxs = np.where(np.diff(np.sign(np.diff(smoothed[section]))) > 0)[0]
                if len(turning_point_idxs) == 0:
                    frame = section[np.argmin(smoothed[section])]
                elif len(turning_point_idxs) == 1:
                    frame = section[turning_point_idxs[0]]
                else:
                    frame = section[turning_point_idxs[i - 1]]
                lims.append(frame)
            bout_lims.append(lims)

    bout_frames = [np.arange(lims[0], lims[1] + 1) for lims in bout_lims if (1 + lims[1] - lims[0]) >= 50]
    bout_frames = [tip.index[frames] for frames in bout_frames]

    return bout_frames


class SetBoutDetectionThreshold(object):
    """Class for setting thresholds for bout detection

    When initialised, a SetBoutDetectionThreshold object will create a matplotlib figure that contains two axes. The
    top axis shows tail tip angle data with detected bouts overlaid. The bottom axis shows filtered differentiated data
    that are used to find the bouts.

    Two values can be adjusted to optimize bout detection. The first - threshold - defines the minimum amount of
    movement/change that has to have occurred between frames for the tail to have been considered to move. The second -
    min length - defines the minimum number of continuous frames that contain movement that must have occurred for the
    section to be labelled as a bout.

    When the window is closed, the set threshold and minimum bout length can be accessed through the threshold and
    min_bout_length attributes respectively.

    Which data that is displayed at any given time can be adjusted using the time and scale sliders. These have no
    impact on bout detection and exist only to assist users in viewing the performance of the algorithm. The time slider
    controls a viewing window that slides over the data and the scale slider adjusts the amount of time shown within
    that window.

    Attributes
    ----------
    User-defined
        data : pd.Series
            The angle of the tip of the tail for all frames in a video (specified by the s parameter)
        fs : float
            The sampling frequency (frames per second, specified by the fs parameter)
        threshold : float
            The threshold for finding bouts (variable, initial value specified by the thresh_init parameter)
        min_bout_length : float
            The minimum duration of a bout (variable, initial value specified by the length_init parameter)


    Generated and updated automatically:
        min_time_scale : 0.1 (seconds)
            The minimum amount of time that can be displayed in the window
        max_time_scale : float
            The maximum amount of time that can be displayed in the window (seconds)
        time_scale : float (starting value = 5)
            The amount of time that is displayed in the window (seconds)
        time_position : float (0-1)
            The position in the recording that is displayed
        n_frames : int or float
            The number of frames that should be displayed in the window
        first_frame : int
            The number of the first frame that should be displayed
        last_frame : int
            The number of the last frame that should be displayed
        fig : plt.Figure
            The figure object
        ax1, ax2 : plt.axes.Axes
            Axes where the tail tip (ax1) and smoothed derivative (ax2) are displayed
        scale_slider_ax, time_slider_ax, thresh_slider_ax, bout_length_slider_ax : matplotlib.axes.Axes
            Slider axes
        scale_slider, time_slider, thresh_slider, bout_length_slider : matplotlib.widgets.Slider
            Slider objects
        window : int
            The number of frames over which a sliding average is applied to find bouts
        filtered : pd.Series
            Smoothed derivative data
        l : matplotlib.lines.Line2D
            Line2D object for plotted filtered data

    See Also
    --------
    find_bouts
    """

    def __init__(self, s, fs=500., thresh_init=0.02, length_init=0.05):
        """__init__ for SetBoutDetectionThreshold class

        Parameters
        ----------
        s : pd.Series
            The angle of the tip of the tail for all frames in a video
        fs : float, optional (default = 500.)
            The sampling frequency (frames per second)
        thresh_init : float, optional (default = 0.02)
            The threshold for finding bouts
        length_init : float, optional (default = 0.05)
            The minimum duration of a bout (seconds)
        """

        self.data = s
        self.fs = fs

        self.min_time_scale = 0.1
        self.max_time_scale = int(len(self.data) / self.fs)

        self.time_scale = 5
        self.time_position = 0

        self.n_frames = self.time_scale * self.fs
        self.first_frame = int((len(self.data) - self.n_frames) * self.time_position)
        self.last_frame = self.first_frame + self.n_frames

        self.fig = plt.figure(figsize=(15, 5))

        self.ax1 = self.fig.add_axes([0.05, 0.5, 0.9, 0.38])
        self.ax2 = self.fig.add_axes([0.05, 0.05, 0.9, 0.38], sharex=self.ax1)
        self.ax1.get_xaxis().set_visible(False)
        self.ax2.set_ylim(0, 0.3)

        self.scale_slider_ax = self.fig.add_axes([0.05, 0.95, 0.9, 0.03])
        self.time_slider_ax = self.fig.add_axes([0.05, 0.9, 0.9, 0.03])
        self.thresh_slider_ax = self.fig.add_axes([0.12, 0.45, 0.3, 0.03])
        self.bout_length_slider_ax = self.fig.add_axes([0.58, 0.45, 0.3, 0.03])

        self.scale_slider = Slider(self.scale_slider_ax, 'scale', self.min_time_scale, self.max_time_scale, self.time_scale)
        self.scale_slider.on_changed(self._update_window)
        self.time_slider = Slider(self.time_slider_ax, 'time', 0, 1, 0)
        self.time_slider.on_changed(self._update_window)

        self.threshold = thresh_init
        self.thresh_slider = Slider(self.thresh_slider_ax, 'threshold', 0, 0.1, self.threshold)
        self.thresh_slider.on_changed(self._update_bouts)

        self.min_bout_length = length_init
        self.bout_length_slider = Slider(self.bout_length_slider_ax, 'min length', 0, 0.5, self.min_bout_length)
        self.bout_length_slider.on_changed(self._update_bouts)

        self.ax1.plot(self.data, c='C0')
        self.filtered, = self.ax2.plot(self.data, c='C0')
        self.l, = self.ax2.plot([0, len(self.data)], [self.threshold, self.threshold], c='k', linestyle='dashed')
        self._update_bouts(0)
        self._update_window(0)

    def _update_bouts(self, val):

        for l in self.ax1.lines[1:]:
            self.ax1.lines.remove(l)

        self.threshold = self.thresh_slider.val
        self.min_bout_length = self.bout_length_slider.val

        window = int(self.min_bout_length * self.fs)
        kernel = gaussian(window * 2, (window * 2) / 5.)
        kernel /= np.sum(kernel)
        diffed = self.data.diff().shift(-1)
        diffed.iloc[-1] = 0
        abs_diffed = diffed.abs()
        filtered = np.convolve(abs_diffed, kernel, mode='same')
        above_threshold = np.where(filtered > self.threshold)[0]
        movement_indices = find_contiguous(above_threshold, minsize=int(self.min_bout_length * self.fs))
        movement_frames = [self.data.index[frame_indices] for frame_indices in movement_indices]

        for frames in movement_frames:
            self.ax1.plot(self.data.loc[frames], c='r')
        self.filtered.set_ydata(filtered)
        self.l.set_ydata([self.threshold, self.threshold])

    def _update_window(self, val):
        """Re-scales the window when either the time scale or time position is changed"""
        self.time_scale = self.scale_slider.val
        self.time_position = self.time_slider.val
        self.n_frames = self.time_scale * self.fs
        self.first_frame = int((len(self.data) - self.n_frames) * self.time_position)
        self.last_frame = self.first_frame + self.n_frames
        self.ax1.set_xlim(self.first_frame,self.last_frame)

    def set(self):
        """Allows thresholds for bout detection to be adjusted

        Returns
        -------
        self.threshold, self.min_bout_length : float
            Threshold and minimum bout length parameters that can be passed to find_bouts
        """
        plt.show()
        return self.threshold, self.min_bout_length


def find_video_bouts(kinematics, fs, threshold=0.02, min_length=0.1):
    """Find bouts within a video using a given threshold and minimum bout length

    This function first finds continuously tracked sections of the video that do not contain any missing data. Next, it
    finds bouts within each continuously tracked segment using the find_bouts function with the given parameters. Then,
    it splits bouts longer than 400 ms into shorter bouts using the split_long_bout function. Finally, it returns the
    frame numbers of each bout that was detected in chronological order.

    Parameters
    ----------
    kinematics : pd.DataFrame or str
        Path to a .csv file containing kinematic data or pre-loaded kinematic data as a DataFrame
    fs : float
        The sampling frequency (frames per second)
    threshold : float, optional (default = 0.02)
        Threshold used for finding bouts
    min_length : float, optional (default = 0.1)
        Minimum bout length (seconds)

    Returns
    -------
    video_bout_frames : list
        A list of tuples containing the first and last frame of each bout

    See Also
    --------
    find_contiguous
    find_bouts
    split_long_bout
    SetBoutDetectionThreshold
    """

    if type(kinematics) == str:
        kinematics_df = pd.read_csv(kinematics)
    else:
        kinematics_df = kinematics

    tracked_data = kinematics_df[kinematics_df['tracked']]
    tip_angle = tracked_data.loc[:, 'tip']

    tracked_segments = find_contiguous(tip_angle.index)

    video_bout_frames = []
    for segment_frames in tracked_segments:
        segment = tip_angle.loc[segment_frames]

        segment_bouts = find_bouts(segment, fs, threshold, min_length)
        segment_bouts_split = []
        for frames in segment_bouts:
            if len(frames) / float(fs) > 0.4:
                split_frames = split_long_bout(segment.loc[frames], fs)
                segment_bouts_split += split_frames
            else:
                segment_bouts_split.append(frames)

        segment_bouts_split.sort(key=lambda frames: frames[0])
        segment_bout_frames = [(frames[0], frames[-1]) for frames in segment_bouts_split if
                               frames[0] != segment_frames[0] and frames[-1] != segment_frames[-1]]

        video_bout_frames += segment_bout_frames

    return video_bout_frames
