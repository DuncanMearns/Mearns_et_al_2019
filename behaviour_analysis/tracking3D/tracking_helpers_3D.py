from ..tracking import *
from ..video import Video
from ..miscellaneous import array2point, find_contiguous, read_csv

import cv2
import numpy as np
import sys
import os
import pandas as pd
import time
from ast import literal_eval


def find_jaw_point(fish_mask, point, vector):
    try:
        line_mask = np.zeros(fish_mask.shape, np.uint8)
        cv2.line(line_mask, array2point(point), array2point(point + 50 * vector), 1, 1)
        points_along_line = np.logical_and(fish_mask, line_mask)
        points_along_line = np.argwhere(points_along_line)[:, ::-1]
        vectors_along_line = points_along_line - point
        distances_along_line = np.linalg.norm(vectors_along_line, axis=1)
        max_dist_index = np.argmax(distances_along_line)
        distance = distances_along_line[max_dist_index]
        jaw_point = points_along_line[max_dist_index]
    except ValueError:  # occurs when there is an error in the tracking and the point is outside of the fish mask
        # just take the nearest point in the fish as the jaw point
        fish_points = np.argwhere(fish_mask)
        delta_vector = fish_points - point[::-1]
        distances = np.linalg.norm(delta_vector, axis=1)
        min_dist_index = np.argmin(distances)
        distance = distances[min_dist_index]
        jaw_point = fish_points[min_dist_index][::-1]
    return jaw_point, distance


def analyse_frame_3d(image, background, top_ROI, side_ROI, thresh1, thresh2, thresh3, thresh4, n_points, return_image=False):
    """Main function for tracking the fish in a video frame

    Parameters
    ----------
    image : np.int8
        8-bit numpy array representing a rasterized color frame from a video

    background : np.int8
        8-bit numpy array representing a background image

    thresh1 : int (0-255)
        The threshold used for finding the fish within the frame

    thresh2 : int (0-255)
        The threshold used for finding the eyes and swim bladder of the fish within the frame

    n_points : int
        The number of points to fit to the tail

    track_tail : bool, optional (default = True)
        Whether to perform tail tracking. Tail tracking is the rate-limiting step in the analysis and does not need to
        be performed if only information about the eyes, heading and position of the fish is being tracked

    return_image : bool, optional (default = False)
        Whether to return the result of the tracking as an image. This should be set to True if the function is being
        called to check the output of tracking (e.g. when setting thresholds) and False when only tracked data is being
        extracted from video frames for saving.

    Returns
    -------
    If return_image = False:
        tracking_params : dict
            Dictionary of tracking data for the frame

    If return_image = True:
        show_contours, show_tracking : np.int8, np.int8
            Two images showing: 1) the contours found using given thresholds; and 2) the final result of the tracking

    See Also
    --------
    track_video
    track_with_watershed
    assign_internal_features
    find_contours
    contour_info
    """
    # BACKGROUND SUBTRACTION
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = background_division(gray, background)

    top_img = crop_to_rectangle(bg, *top_ROI)
    scaled = cv2.resize(top_img, None, fx=2, fy=2)

    # FIND FISH
    top_contours = find_contours(scaled, thresh1)
    mask = np.zeros(scaled.shape, np.uint8)
    masked = mask.copy()
    cv2.drawContours(mask, top_contours, 0, 1, -1)
    mask = mask.astype(np.bool)
    masked[mask] = scaled[mask]
    masked = cv2.equalizeHist(masked)

    # FIND INTERNAL CONTOURS
    internal_contours = find_contours(masked, thresh2)[:3]
    try:
        if len(internal_contours) == 3:  # eye tracking works
            centres, angles = zip(*[contour_info(cntr) for cntr in internal_contours])
            centres, angles = np.array(centres), np.array(angles)
            internal_features = assign_internal_features(centres, angles)
            if not 20 <= internal_features['inter_eye_angle'] <= 40:
                centres, angles = track_with_watershed(masked, internal_contours)
                internal_features = assign_internal_features(centres, angles)
        elif len(internal_contours) > 0:
            centres, angles = track_with_watershed(masked, internal_contours)
            internal_features = assign_internal_features(centres, angles)
        else:
            raise TrackingError()

        sb_c, left_c, right_c = internal_features['sb_c'], internal_features['left_c'], internal_features['right_c']
        left_th, right_th = internal_features['left_th'], internal_features['right_th']

        eye_midpoint = np.mean([left_c, right_c], axis=0)
        heading_vector = eye_midpoint - sb_c
        heading_vector /= np.linalg.norm(heading_vector)
        heading = np.arctan2(*heading_vector[::-1])

        left_v = np.array([np.cos(left_th), np.sin(left_th)])
        right_v = np.array([np.cos(right_th), np.sin(right_th)])
        left_dot, right_dot = np.dot(left_v, heading_vector), np.dot(right_v, heading_vector)
        if left_dot == 0 or right_dot == 0:
            raise TrackingError()
        if left_dot < 0:
            left_th += np.pi
            left_v *= -1
        if right_dot < 0:
            right_th += np.pi
            right_v *= -1

        # TAIL TRACKING
        tail_points = fit_tail(mask, sb_c, heading_vector, n_points)

        # SIDE TRACKING
        side_img = crop_to_rectangle(bg, *side_ROI)
        scaled = cv2.resize(side_img, None, fx=2, fy=2)
        if abs(np.tan(heading)) < 1:  # if fish is within 45 degrees of the imaging plane
            x_direction = np.sign(np.cos(heading))
            # FIND FISH
            side_contours = find_contours(scaled, thresh3)
            mask = np.zeros(scaled.shape, np.uint8)
            masked = mask.copy()
            try:  # usually fish contour is the largest
                side_contour = side_contours[0]
                cv2.drawContours(mask, [side_contour], 0, 1, -1)
                mask = mask.astype(np.bool)
                masked[mask] = scaled[mask]
                masked = cv2.equalizeHist(masked)
                # FIND HEAD
                head_contours = find_contours(masked, thresh4)
                head_contour = head_contours[0]
            except IndexError:  # this occurs if there is no head contour, in which case fish contour is probably wrong
                side_contour = side_contours[1]
                cv2.drawContours(mask, [side_contour], 0, 1, -1)
                mask = mask.astype(np.bool)
                masked[mask] = scaled[mask]
                masked = cv2.equalizeHist(masked)
                # FIND HEAD
                head_contours = find_contours(masked, thresh4)
                head_contour = head_contours[0]
            # CALCULATE HEAD DIRECTION AND TAIL ANGLE
            head_centre, head_direction = contour_info(head_contour)
            head_vector = np.array([np.cos(head_direction), np.sin(head_direction)])
            if np.sign(head_vector[0]) == -x_direction:
                head_direction += np.pi
                head_vector *= -1
            side_centre, tail_angle = contour_info(side_contour)
            tail_vector = np.array([np.cos(tail_angle), np.sin(tail_angle)])
            if np.sign(np.dot(head_vector, tail_vector)) == 1:
                tail_angle += np.pi
                tail_vector *= -1

            # CALCULATE JAW PARAMETERS
            mandible, protrusion = find_jaw_point(mask, head_centre, head_vector)
            head_midpoint = head_centre + (0.5 * protrusion * head_vector)
            R = np.array([[0, -x_direction], [x_direction, 0]])
            normal_vector = np.dot(R, head_vector)
            hyoid, depression = find_jaw_point(mask, head_midpoint, normal_vector)
            side_tracked = True

        else:
            side_tracked = False

        if return_image:
            top_img = cv2.resize(crop_to_rectangle(gray, *top_ROI), None, fx=2, fy=2)
            side_img = cv2.resize(crop_to_rectangle(gray, *side_ROI), None, fx=2, fy=2)
            tracking_img = cv2.resize(image, None, fx=2, fy=2)
            colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255), m=(255, 0, 255))
            # draw contours
            cv2.drawContours(top_img, top_contours, 0, colors['k'], 1)
            cv2.drawContours(top_img, internal_contours, -1, colors['w'], 1)
            if side_tracked:
                cv2.drawContours(side_img, [side_contour], 0, colors['k'], 1)
                cv2.drawContours(side_img, [head_contour], 0, colors['w'], 1)
            # draw tracked points
            # plot heading
            top_shift = np.array(top_ROI[0]) * 2
            side_shift = np.array(side_ROI[0]) * 2
            cv2.circle(tracking_img, array2point(sb_c + top_shift), 3, colors['y'], -1)
            cv2.line(tracking_img, array2point(sb_c + top_shift), array2point(sb_c + (80 * heading_vector) + top_shift), colors['y'], 2)
            if side_tracked:
                # plot fish elevation
                cv2.line(tracking_img, array2point(side_centre + side_shift), array2point(side_centre + (100 * tail_vector) + side_shift), colors['b'], 2)
                # plot cranial elevation
                cv2.circle(tracking_img, array2point(side_centre + side_shift), 3, colors['y'], -1)
                cv2.line(tracking_img, array2point(side_centre + side_shift), array2point(side_centre + (50 * head_vector) + side_shift), colors['y'], 2)
                # plot hyoid depression
                cv2.circle(tracking_img, array2point(head_midpoint + side_shift), 3, colors['m'], -1)
                cv2.line(tracking_img, array2point(head_midpoint + side_shift), array2point(hyoid + side_shift), colors['m'], 2)
            # plot eyes
            for p, v, c in zip((left_c, right_c), (left_v, right_v), ('g', 'r')):
                cv2.line(tracking_img, array2point(p - (20 * v) + top_shift), array2point(p + (20 * v) + top_shift), colors[c], 2)
            # plot tail points
            for p in tail_points:
                cv2.circle(tracking_img, array2point(p + top_shift), 1, colors['b'], -1)

            return top_img, side_img, tracking_img

        tracking_params = {'centre': tuple(sb_c), 'heading': heading,
                           'left_centre': tuple(left_c), 'left_angle': left_th,
                           'right_centre': tuple(right_c), 'right_angle': right_th,
                           'tail_points': tail_points,
                           'tracked': True}
        if side_tracked:
            tracking_params['side_tracked'] = True
            tracking_params['side_centre'] = tuple(side_centre)
            tracking_params['fish_elevation'] = tail_angle
            tracking_params['head_midpoint'] = tuple(head_midpoint)
            tracking_params['head_elevation'] = head_direction
            tracking_params['hyoid'] = tuple(hyoid)
            tracking_params['depression'] = depression
        else:
            tracking_params['side_tracked'] = False
            param_names = ['side_centre', 'fish_elevation', 'head_midpoint', 'head_elevation', 'hyoid', 'depression']
            for param in param_names:
                tracking_params[param] = None

    except TrackingError:

        if return_image:
            top_img = cv2.resize(crop_to_rectangle(gray, *top_ROI), None, fx=2, fy=2)
            side_img = cv2.resize(crop_to_rectangle(gray, *side_ROI), None, fx=2, fy=2)
            tracking_img = cv2.resize(image, None, fx=2, fy=2)
            # draw contours
            cv2.drawContours(top_img, top_contours, 0, (0, 0, 0), 1)
            cv2.drawContours(top_img, internal_contours, -1, (255, 255, 255), 1)
            return top_img, side_img, tracking_img

        param_names = ['centre', 'heading', 'left_centre', 'left_angle', 'right_centre', 'right_angle',
                       'side_centre', 'fish_elevation', 'head_midpoint', 'head_elevation', 'hyoid', 'depression']
        tracking_params = dict([(param_name, None) for param_name in param_names])
        tracking_params['tail_points'] = np.zeros((n_points, 2)) * np.nan
        tracking_params['tracked'] = False
        tracking_params['side_tracked'] = False

    return tracking_params


def set_thresholds_3d(video_paths, background_paths, top_ROI, side_ROI, thresh1_initial=10, thresh2_initial=210, thresh3_initial=20, thresh4_initial=200, n_points=51):
    """Set the thresholds for finding the fish and internal contours in videos

    Creates windows to set thresholds for tracking.
    The 'set thresholds' window contains track bars for adjusting the frame number and threshold values.
    The 'contours' window displays contours that were found using the current thresholds.
    The 'tracking' window displays the final output of the tracking obtained using the current thresholds.

    First, thresh1 should be adjusted so that the fish is properly outlined in the 'contours' window.
    Then, thresh2 should be adjusted so that the eyes and swim bladder are outlined in the majority of frames.

    Pressing the space key opens the next video.
    Pressing the enter key returns the current thresholds.
    Pressing the escape key quits using sys.exit().

    Parameters
    ----------
    video_paths : list or tuple
        List of paths to video files (.avi) that should all be tracked using the same thresholds

    background_path : str
        Path to a background image

    thresh1_initial : int, optional (0-255, default = 10)
        The initial threshold to use for finding the fish

    thresh2_initial : int, optional (0-255, default = 210)
        The initial threshold to use for finding the eyes and swim bladder of the fish

    n_points : int, optional (default = 51)
        Number of points to fit to the tail

    Returns
    -------
    thresh1, thresh2 : int
        Values for thresholds to use for tracking

    Notes
    -----
    For optimal tracking, it is best to use lower thresholds. If tail tracking is going to be applied, thresh1 should be
    set low enough such that the tail is successfully tracked to the very tip. Tail tracking is robust to contours that
    are slightly larger than the true outline of the fish. However, if this threshold is too low then tail tracking
    might fail if the fish does a large turn and the head "fuses" with the tail.

    Thresh1 should always be set first, as eye tracking depends on applying the thresholds sequentially and thus is
    sensitive to both. Thresh 2 should be set such that it properly outlines the eyes and swim bladder as three distinct
    contours in the majority of frames. However, it should not be excessively high to properly separate contours in all
    frames, since the tracking algorithm applies a watershed when contours fuse. The watershed can be less accurate and
    slower than tracking when there are three contours, but works well when there are only a few frames where contours
    briefly fuse.
    """
    thresh1, thresh2, thresh3, thresh4 = thresh1_initial, thresh2_initial, thresh3_initial, thresh4_initial
    key_input = KeyboardInteraction()
    for video_path, background_path in zip(video_paths, background_paths):
        v = Video(video_path)
        background = cv2.imread(background_path, 0)
        # setup the windows
        cv2.namedWindow('tracking')
        cv2.namedWindow('top contours')
        cv2.namedWindow('side contours')
        for window in ['tracking', 'top contours', 'side contours']:
            cv2.createTrackbar('frame', window, 0, v.frame_count - 1, v.frame_change)
        cv2.createTrackbar('thresh1', 'top contours', thresh1, 255, lambda x: x)
        cv2.createTrackbar('thresh2', 'top contours', thresh2, 255, lambda x: x)
        cv2.createTrackbar('thresh3', 'side contours', thresh3, 255, lambda x: x)
        cv2.createTrackbar('thresh4', 'side contours', thresh4, 255, lambda x: x)
        while True:
            for window in ['tracking', 'top contours', 'side contours']:
                cv2.setTrackbarPos('frame', window, v.frame_number)
            thresh1 = cv2.getTrackbarPos('thresh1', 'top contours')
            thresh2 = cv2.getTrackbarPos('thresh2', 'top contours')
            thresh3 = cv2.getTrackbarPos('thresh3', 'side contours')
            thresh4 = cv2.getTrackbarPos('thresh4', 'side contours')

            frame = v.grab_frame()
            top_contours, side_contours, tracking = analyse_frame_3d(frame, background, top_ROI, side_ROI, thresh1, thresh2, thresh3, thresh4, n_points=n_points, return_image=True)
            cv2.imshow('top contours', top_contours)
            cv2.imshow('side contours', side_contours)
            cv2.imshow('tracking', tracking)

            k = cv2.waitKey(1)
            if key_input.valid(k):
                break

        if k == key_input.enter:
            cv2.destroyAllWindows()
            break
        elif k == key_input.esc:
            cv2.destroyAllWindows()
            sys.exit()
        else:
            pass

    return thresh1, thresh2, thresh3, thresh4


def track_video_3d(video_path, background_path, top_ROI, side_ROI, thresh1, thresh2, thresh3, thresh4, n_points, save_output=False, filename=None, output_directory=None):
    """Main function for tracking all frames in a video

    This function either returns a DataFrame and an array (if save_output = False), or it saves the output of the
    tracking as two files (filename.csv and filename.npy) in the output_directory and returns the time it took to track
    the video (if save_output = True).

    Parameters
    ----------
    video_path : str
        Path to a video file

    background : np.int8
        8-bit numpy array representing a background image

    thresh1 : int (0-255)
        The threshold used for finding the fish

    thresh2 : int (0-255)
        The threshold used for finding the eyes and swim bladder of the fish

    n_points : int
        The number of points to fit to the tail

    track_tail : bool, optional (default = True)
        Whether to perform tail tracking. Tail tracking is the rate-limiting step in the analysis and does not need to
        be performed if only information about the eyes, heading and position of the fish is being tracked

    save_output : bool, optional (default = False)
        Whether to save the output of the tracking

    filename : str or None, optional (default = None)
        The basename for saving tracking files if save_output = True

    output_directory : str or None, optional (default = None)
        The output directory for saving tracking files if save_output = True


    Returns
    -------
    if save_output = False:
        tracking_df, tail_points : pd.DataFrame, np.array
            The output of the tracking. The tracking_df, shape = (number of frames, number of features), contains
            information about the position and orientation of the fish and its eyes in each frame. The tail_points
            array, shape = (number of frames, number of points, 2) contains points along the tail in each frame as xy
            coordinates.
    if save_output = True:
        tracking_time : float
            The time it took to analyse the video (seconds).

    Notes
    -----
    The tracking_df contains the columns:
        'centre', 'heading', 'left_centre', 'left_angle', 'right_centre', 'right_angle', 'tracked'

    See Also
    --------
    analyse_frame
    """
    start_tracking_time = time.time()
    cap = cv2.VideoCapture(video_path)
    background = cv2.imread(background_path, 0)

    tracking_cols = ['centre', 'heading', 'left_centre', 'left_angle', 'right_centre', 'right_angle', 'tracked',
                     'side_centre', 'fish_elevation', 'head_midpoint', 'head_elevation', 'hyoid', 'depression', 'side_tracked']

    tracking_dict = dict([(col, []) for col in tracking_cols])
    tail_points = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame_info = analyse_frame_3d(frame, background, top_ROI, side_ROI, thresh1, thresh2, thresh3, thresh4, n_points)
            for col in tracking_cols:
                tracking_dict[col].append(frame_info[col])
            tail_points.append(frame_info['tail_points'])
        else:
            break
    tracking_df = pd.DataFrame(tracking_dict, columns=tracking_cols)
    tail_points = np.array(tail_points, dtype=np.float32)
    tracking_time = time.time() - start_tracking_time
    if save_output:
        if filename is None:
            output_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            output_name = filename
        if output_directory is None:
            tracking_path = output_name + '.csv'
            tail_points_path = output_name + '.npy'
        else:
            tracking_path = os.path.join(output_directory, output_name + '.csv')
            tail_points_path = os.path.join(output_directory, output_name + '.npy')
        tracking_df.to_csv(tracking_path, index=False)
        np.save(tail_points_path, tail_points)
        return tracking_time
    else:
        return tracking_df, tail_points


def show_tracking_3d(frame, top_ROI, side_ROI,
                     centre, heading, left_centre, left_angle, right_centre, right_angle, points, tracked,
                     side_centre, fish_elevation, head_midpoint, head_elevation, hyoid, depression, side_tracked):
    """Draws tracked points and angles onto an image

    Parameters
    ----------
    frame : np.uint8
        Unsigned 8-bit numpy array representing a rasterised color frame from a video

    centre : tuple (x, y)
        The centre of the fish

    heading : float
        The angle of the heading of the fish (radians)

    left_centre : tuple (x, y)
        The centre of the left eye

    left_angle : float
        The angle of the left eye (radians)

    right_centre : tuple (x, y)
        The centre of the right eye

    right_angle : float
        The angle of the right eye (radians)

    points : np.array, shape: (n_points, 2)
        Array representing points (as xy coordinates) along the fish's tail

    tracked : bool
        Whether the fish has been successfully tracked in the frame

    Returns
    -------
    img : np.uint8
        Re-scaled frame showing the tracked points and angles
    """
    img = cv2.resize(frame, None, fx=2, fy=2)
    colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255), m=(255, 0, 255))
    top_shift = np.array(top_ROI[0]) * 2
    side_shift = np.array(side_ROI[0]) * 2
    # draw tracked points
    if tracked:
        # plot heading
        heading_vector = np.array([np.cos(heading), np.sin(heading)])
        cv2.circle(img, array2point(top_shift + centre), 3, colors['y'], -1)
        cv2.line(img, array2point(top_shift + centre), array2point(top_shift + centre + (80 * heading_vector)), colors['y'], 2)
        # plot eyes
        left_v = np.array([np.cos(left_angle), np.sin(left_angle)])
        right_v = np.array([np.cos(right_angle), np.sin(right_angle)])
        for p, v, c in zip((left_centre, right_centre), (left_v, right_v), ('g', 'r')):
            cv2.line(img, array2point(top_shift + p - (20 * v)), array2point(top_shift + p + (20 * v)), colors[c], 2)
        # plot tail points
        for p in points:
            cv2.circle(img, array2point(top_shift + p), 1, colors['b'], -1)
        if side_tracked:
            # plot fish elevation
            tail_vector = np.array([np.cos(fish_elevation), np.sin(fish_elevation)])
            cv2.line(img, array2point(side_shift + side_centre), array2point(side_shift + side_centre + (100 * tail_vector)), colors['b'], 2)
            # plot cranial elevation
            head_vector = np.array([np.cos(head_elevation), np.sin(head_elevation)])
            cv2.circle(img, array2point(side_shift + side_centre), 3, colors['y'], -1)
            cv2.line(img, array2point(side_shift + side_centre), array2point(side_shift + side_centre + (50 * head_vector)), colors['y'], 2)
            # plot hyoid depression
            cv2.circle(img, array2point(side_shift + head_midpoint), 3, colors['m'], -1)
            cv2.line(img, array2point(side_shift + head_midpoint), array2point(side_shift + hyoid), colors['m'], 2)
    return img


def check_tracking_3d(video, tracking, points, top_ROI, side_ROI, **kwargs):
    """Check the tracking for a video

    Parameters
    ----------
    video : str or Video
        Path to an avi file or a Video object

    tracking : str or pd.DataFrame
        Path to a csv file or a DataFrame containing tracking information for the video

    points : str or np.array
        Path to a npy file or an array containing tail points in each frame of the video

    kwargs : dict, optional
        winname : the name of window in which to display frames (default = 'check tracking')

    Returns
    -------
    v : Video
        Video object
    """
    # Get the video, tracking and points objects
    if type(video) == str:
        v = Video(video)
    else:
        v = video
    if type(tracking) == str:
        tracking_df = read_csv(tracking, centre=literal_eval, left_centre=literal_eval, right_centre=literal_eval,
                               side_centre=literal_eval, head_midpoint=literal_eval, hyoid=literal_eval)
    else:
        tracking_df = tracking
    if type(points) == str:
        tail_points = np.load(points)
    else:
        tail_points = points
    # Check tracking
    if 'winname' in kwargs.keys():
        winname = kwargs['winname']
    else:
        winname = 'check tracking'
    cv2.namedWindow(winname)
    cv2.createTrackbar('frame', winname, 0, v.frame_count - 1, v.frame_change)
    while True:
        frame = v.grab_frame()
        tracking_kwargs = tracking_df.loc[v.frame_number]
        image = show_tracking_3d(frame, top_ROI, side_ROI, points=tail_points[v.frame_number], **tracking_kwargs)
        cv2.imshow(winname, image)
        v.wait(1)
        if v.valid():
            cv2.destroyWindow(winname)
            break
    return v


def calculate_kinematics_3d(tracking_path, points_path, fs, min_tracked_length=0.1, smooth_eye_angles=True, smooth_tail=True,
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

    start_analysis_time = time.time()

    # IMPORT TAIL POINTS AND KINEMATICS
    tail_points = np.load(points_path)
    tracking_df = read_csv(tracking_path, centre=literal_eval)  #, left_centre=literal_eval, right_centre=literal_eval)

    # CREATE KINEMATICS DATAFRAME
    k_cols = ['k{}'.format(i) for i in range(tail_points.shape[1] - 1)]
    n_tip_columns = int(len(k_cols) / 5)
    tip_columns = k_cols[-n_tip_columns:]

    columns = k_cols + ['tip', 'length', 'left', 'right', 'speed', 'angular_velocity', 'tracked', 'depression', 'elevation', 'fish_elevation', 'side_tracked']
    kinematics = pd.DataFrame(index=tracking_df.index, columns=columns)
    kinematics['tracked'] = False
    kinematics['side_tracked'] = False

    # FIND TRACKED SEGMENTS
    tracked_frames = tracking_df[tracking_df['tracked']]
    tracked_segments = find_contiguous(tracked_frames.index, 1, int(min_tracked_length * fs))

    for segment_frames in tracked_segments:
        first, last = segment_frames[0], segment_frames[-1]
        segment = tracking_df.loc[first:last, :].copy()  # make a copy of the tracked segment
        for col in ['heading', 'left_angle', 'right_angle']:
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

        # SIDE KINEMATICS
        tracked_side_frames = segment[segment['side_tracked']]
        tracked_side_segments = find_contiguous(tracked_side_frames.index, 1, int(min_tracked_length * fs))
        for side_segment_frames in tracked_side_segments:
            first, last = side_segment_frames[0], side_segment_frames[-1]
            side_segment = segment.loc[first:last].copy()
            # Fish and cranial elevation
            head_elevation = side_segment['head_elevation']
            fish_elevation = side_segment['fish_elevation']
            v, u = np.array([np.cos(head_elevation), np.sin(head_elevation)]), np.array([np.cos(fish_elevation), np.sin(fish_elevation)])
            elevation = np.arcsin(np.cross(v.T, u.T))
            elevation *= np.sign(u[0])
            fish_elevation = np.arcsin(u[1])
            side_segment.loc[:, 'elevation'] = elevation
            # Low pass filtering of hyoid depression and cranial elevation
            jaw = side_segment.loc[first:last, ['depression', 'elevation']].copy()
            low_pass = jaw.copy()
            jaw = jaw.rolling(window=5, min_periods=0, center=True, axis=0).median()
            low_pass = low_pass.rolling(window=int(0.25 * fs), min_periods=0, center=True, axis=0).min()
            low_pass = low_pass.rolling(window=int(fs), min_periods=0, center=True, axis=0).mean()
            # Update kinematics DataFrame
            kinematics.loc[first:last, 'fish_elevation'] = fish_elevation
            kinematics.loc[first:last, ['depression', 'elevation']] = jaw - low_pass
            kinematics.loc[first:last, 'side_tracked'] = True

    analysis_time = time.time() - start_analysis_time

    if save_output:
        if output_path is None:
            save_path = os.path.splitext(os.path.basename(tracking_path))[0] + '_kinematics.csv'
        else:
            save_path = output_path
        kinematics.to_csv(save_path, index=False)
        return analysis_time
    else:
        return kinematics
