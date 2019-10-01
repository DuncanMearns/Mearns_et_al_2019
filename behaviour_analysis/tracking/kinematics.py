from ..miscellaneous import read_csv, find_contiguous, Timer

import numpy as np
from scipy.signal import gaussian
from ast import literal_eval
import pandas as pd
import os


def calculate_eye_angles(eye_angles, heading_angles):
    """Calculate the angles of an eye relative to the heading

    First, the angles are converted into unit vectors. For each frame the cross product is calculated between the eye
    vector and the heading vector. The signed angle between the heading vector and eye vector is the arcsin of their
    cross product: a x b = |a||b|sin(theta) -> theta = arcsin(a x b)

    Parameters
    ----------
    eye_angles, heading_angles : pd.Series
        The angle of an eye and the heading over many frames (angles should be in radians)

    Returns
    -------
    corrected_eye_angles : np.array (ndim = 1)
        The corrected angles of the eye for all frames
    """
    eye_vectors = np.array([np.cos(eye_angles), np.sin(eye_angles)])
    heading_vectors = np.array([np.cos(heading_angles), np.sin(heading_angles)])
    corrected_eye_angles = np.arcsin(np.cross(eye_vectors.T, heading_vectors.T))
    return corrected_eye_angles


def smooth_tail_points(points, size=7, kind='boxcar'):
    """Smooths the position of points along the tail

    Parameters
    ----------
    points : array-like
        An array representing the tail points in each frame, shape (n_frames, n_points, 2)
    size : int, optional
        The width of the filter (number of points to average)
    kind : str {'boxcar', 'gaussian'}, optional
        The shape of the kernel for smoothing points (i.e. how points should be weighted)

    Returns
    -------
    smoothed_points : ndarray
        The smoothed positions of tail points (same shape as input array)

    Notes
    -----
    This function returns smooths points spatially, not temporally
    """
    if kind == 'boxcar':
        kernel = np.ones((size,))
    elif kind == 'gaussian':
        kernel = gaussian(size, 1)
    else:
        raise ValueError()
    kernel /= np.sum(kernel)
    n = (size - 1) / 2
    padded_points = np.pad(points, ((0, 0), (n, n), (0, 0)), 'edge')
    smoothed_points = np.apply_along_axis(np.convolve, 1, padded_points, kernel, mode='valid')
    return smoothed_points


def calculate_tail_curvature(points, headings):
    """Converts tail points to a 1D vector of angles for each frame after smoothing with gaussian filter

    In each frame, the tail is essentially rotated and centred such that first point is at the origin and the fish
    is facing right along the x-axis. Then, tangents are approximated between pairs of adjacent points. The shape of the
    tail is defined by the angles at which each of these tangents intersect the x-axis.

    Parameters
    ----------
    points : np.array
        Array representing the position of points along the tail within an image, shape = (n_frames, n_points, 2)
    headings : np.array
        Array representing the heading of the fish in each frame (radians), shape = (n_frames,)

    Returns
    -------
    ks, tail_lengths : np.array
        Tail angles and length of the tail in each frame

    Notes
    -----
    Since tail angles are calculated from tangents drawn between successive pairs of points in each frame, the length of
    the vector that defines the shape of the tail will be one less than the number of points fitted to the tail (i.e.
    fitting 51 points to the tail yields a 50 dimensional vector that describes its shape).

    Representing the tail this way requires the heading to the accurately known and points to be equally spaced along
    the tail.
    """
    headings_r = headings + np.pi
    smoothed_points = smooth_tail_points(points, size=7, kind='gaussian')

    vs = np.empty(smoothed_points.shape)
    vs[:, 0] = np.array([np.cos(headings_r), np.sin(headings_r)]).T
    vs[:, 1:] = np.diff(smoothed_points, axis=1)

    ls = np.linalg.norm(vs, axis=2)
    crosses = np.cross(vs[:, :-1], vs[:, 1:])
    crosses /= (ls[:, :-1] * ls[:, 1:])
    dks = np.arcsin(crosses)
    ks = np.cumsum(dks, axis=1)

    tail_lengths = np.sum(ls[:, 1:], axis=1)

    return ks, tail_lengths


def calculate_speed(centres, fs, scale):
    """Calculates the instantaneous speed in each frame

    Parameters
    ----------
    centres : pd.Series
        The centre of the fish (x, y) coordinate in each frame
    fs : float
        The sampling frequency of the data (frames per second)
    scale : float
        The image scale (the size of one pixel)

    Returns
    -------
    speeds : np.array
        The instantaneous speed in each frame (note: length wil be one less than the input)
    """
    xs = centres.apply(lambda c: c[0])
    ys = centres.apply(lambda c: c[1])
    positions = np.array([xs, ys]).T
    vectors = np.diff(positions, axis=0)
    distances = np.linalg.norm(vectors, axis=1)
    speeds = distances * float(scale) * float(fs)
    return speeds


def calculate_angular_velocity(headings, fs):
    """Calculate the instantaneous angular velocity in each frame

    Parameters
    ----------
    headings : pd.Series or np.array
        The heading in each frame (radians)
    fs : float
        The sampling frequency of the data (frames per second)

    Returns
    -------
    angular_velocity : np.array
        The instantaneous angular velocity in each frame (note: length will be one less than the input)
    """
    heading_vectors = np.array([np.cos(headings), np.sin(headings)]).T
    sin_angular_change = np.cross(heading_vectors[:-1], heading_vectors[1:])
    angular_velocity = np.arcsin(sin_angular_change) * float(fs)
    return angular_velocity


def calculate_kinematics(tracking_path, points_path, fs, min_tracked_length=0.1, smooth_eye_angles=True, smooth_tail=True,
                         save_output=False, output_path=None):
    """Generate kinematic data for a tracked video

    Parameters
    ----------
    tracking_path : str
        Path to a .csv file containing tracking data

    points_path : str
        Path to a .npy file containing tail points

    fs : float
        Sampling frequency (frames per second)

    min_tracked_length : float, optional (default = 0.1)
        The minimum length of time for which continuously tracked data are available to calculate kinematics (seconds)

    smooth_eye_angles : bool, optional (default = True)
        Whether to apply a 100 ms sliding median to the eye angle data (edge-preserving smoothing of eye angles)

    smooth_tail : bool, optional (default = True)
        Whether to apply a 3 frame sliding median to the tail tracking data (removes single frame noise from tracking)

    save_output : bool, optional (default = False)
        Whether to save the output

    output_path : str or None, optional (default = None)
        The output path (.csv) for saving kinematics if save_output = True

    Returns
    -------
    if save_output = False:
        kinematics : pd.DataFrame
            DataFrame containing kinematic data. Columns:
                'k0' - 'k(n-1)' : angle of tangents between n successive tail points
                'tip' : the average curvature (tail angle) over the last 20% of the tail
                'length' : the length of the tail (useful for finding tracking errors)
                'left' : the angle of the left eye relative to the heading
                'right' : the angle of the right eye relative to the heading
                'speed' : the instantaneous speed in each frame
                'angular_velocity' : the instantaneous angular velocity in each frame
                'tracked' : whether kinematic data exists from the frame
    if save_output = True:
        analysis_time : float
            The time it took to perform the analysis (seconds).

    """

    timer = Timer()
    timer.start()

    # IMPORT TAIL POINTS AND KINEMATICS
    tail_points = np.load(points_path)
    try:
        tracking_df = read_csv(tracking_path, centre=literal_eval, left_centre=literal_eval, right_centre=literal_eval)
        eyes_tracked = True
    except KeyError:
        tracking_df = read_csv(tracking_path, centre=literal_eval)
        eyes_tracked = False

    # CREATE KINEMATICS DATAFRAME
    k_cols = ['k{}'.format(i) for i in range(tail_points.shape[1] - 1)]
    n_tip_columns = int(len(k_cols) / 5)
    tip_columns = k_cols[-n_tip_columns:]

    if eyes_tracked:
        columns = k_cols + ['tip', 'length', 'left', 'right', 'speed', 'angular_velocity', 'tracked']
    else:
        columns = k_cols + ['tip', 'length', 'speed', 'angular_velocity', 'tracked']
    kinematics = pd.DataFrame(index=tracking_df.index, columns=columns)
    kinematics['tracked'] = False

    # FIND TRACKED SEGMENTS
    tracked_frames = tracking_df[tracking_df['tracked']]
    tracked_segments = find_contiguous(tracked_frames.index, 1, int(min_tracked_length * fs))

    for segment_frames in tracked_segments:
        first, last = segment_frames[0], segment_frames[-1]
        segment = tracking_df.loc[first:last, :].copy()  # make a copy of the tracked segment
        segment.loc[:, 'heading'] = segment.loc[:, 'heading'].rolling(window=3, min_periods=0, center=True).median()
        if eyes_tracked:
            for col in ['left_angle', 'right_angle']:
                segment.loc[:, col] = segment.loc[:, col].rolling(window=3, min_periods=0, center=True).median()

            # CALCULATE EYE ANGLES
            left_angles = calculate_eye_angles(segment.loc[:, 'left_angle'], segment.loc[:, 'heading'])
            right_angles = calculate_eye_angles(segment.loc[:, 'right_angle'], segment.loc[:, 'heading'])
            kinematics.loc[first:last, 'left'] = left_angles
            kinematics.loc[first:last, 'right'] = right_angles
            # SMOOTH EYE ANGLES
            if smooth_eye_angles:
                window = int(0.1 * fs)  # 100ms rolling median
                for col in ['left', 'right']:
                    kinematics.loc[first:last, col] = kinematics.loc[first:last, col].rolling(window=window,
                                                                                              min_periods=0,
                                                                                              center=True).median()

        # ANALYSE TAIL
        points = tail_points[first:last + 1]
        # CALCULATE TAIL ANGLES
        heading = segment.loc[first:last, 'heading']
        ks, tail_lengths = calculate_tail_curvature(points, heading)

        kinematics.loc[first:last, k_cols] = np.array(ks)
        if smooth_tail:
            kinematics.loc[first:last, k_cols] = kinematics.loc[:, k_cols].rolling(window=3, min_periods=0,
                                                                                   center=True).median()
        kinematics.loc[first:last, 'tip'] = kinematics.loc[first:last, tip_columns].apply(np.mean, axis=1)
        kinematics.loc[first:last, 'length'] = tail_lengths

        # CALCULATE HIGHER ORDER KINEMATICS
        speed = calculate_speed(segment['centre'], fs, 1)
        angular_velocity = calculate_angular_velocity(segment['heading'], fs)
        kinematics.loc[first:last - 1, 'speed'] = speed
        kinematics.loc[first:last - 1, 'angular_velocity'] = angular_velocity

        # SET TRACKED TO TRUE IN KINEMATICS
        kinematics.loc[first:last, 'tracked'] = True

    analysis_time = timer.stop()

    if save_output:
        if output_path is None:
            save_path = os.path.splitext(os.path.basename(tracking_path))[0] + '_kinematics.csv'
        else:
            save_path = output_path
        kinematics.to_csv(save_path, index=False)
        return analysis_time
    else:
        return kinematics
