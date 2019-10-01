from .tail_fit import fit_tail
from .background import background_division
from ..miscellaneous import array2point, Timer

import cv2
import numpy as np
import pandas as pd
import os


class TrackingError(Exception):
    """An error that can be raised if any problem is detected in tracking"""
    def __init__(self):
        super(self.__class__, self).__init__()


def contour_info(contour):
    """Uses image moments to find the centre and orientation of a contour

    Parameters
    ----------
    contour : array like
        A contour represented as an array

    Returns
    -------
    c, theta : array, float
        The centre of the contour and its orientation (radians, -pi/2 < theta <= pi/2)
    """
    moments = cv2.moments(contour)
    try:
        c = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    except ZeroDivisionError:
        c = np.mean(contour, axis=0)
        c = tuple(c.squeeze())
    theta = 0.5 * np.arctan2(2 * moments["nu11"], (moments["nu20"] - moments["nu02"]))
    return c, theta


def find_contours(image, thresh):
    """Finds all the contours in an image after binarising with a specified threshold

    Parameters
    ----------
    image : array like
        Unsigned 8-bit integer array representing an image
    thresh : int (0-255)
        Threshold applied to image before finding contours (image is binarised using this threshold)

    Returns
    -------
    contours : list
        A list of arrays representing all the contours found in the image sorted by contour area (largest first)
    """
    ret, threshed = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    img, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
    return contours


def assign_internal_features(centres, angles):
    """Assigns points and angles to the left eye, right eye and swim bladder

    The order of the centres and angles should be the same (centres[0] and angles[0] should correspond to the same contour)

    Parameters
    ----------
    centres : array-like, shape (3, 2)
        Three points representing the centres of the eyes and swim bladder
    angles : array-like, shape (3,)
        Three angles representing the angles of the eyes and swim bladder

    Returns
    -------
    internal_features : dict
        Dictionary of values for the centres and angles of the eyes and swim bladder
            sb_c, sb_th : the centre and angle of the swim bladder
            left_c, left_th : the centre and angle of the left eye
            right_c, right_th : the centre and angle of the right eye
            inter_eye_angle : the angle in between the vectors sb_c -> left_c and sb_c -> right_c
    """
    # create empty array to store distances between points
    distances = np.empty((3,))
    # calculate distances between all pairs of points
    for i, j, k in ((0, 0, 1), (1, 0, 2), (2, 1, 2)):
        distances[i] = np.linalg.norm(centres[k] - centres[j])
    # assign the swim bladder index (i.e. the point that is furthest away from the other two)
    sb_index = 2 - distances.argmin()
    # assign the eye indices
    eye_indices = [i for i in range(3) if i != sb_index]
    # assign centres and angles to either the swim bladder or eyes
    sb_c, sb_th = centres[sb_index], angles[sb_index]
    eye_centres = centres[eye_indices]
    eye_angles = angles[eye_indices]
    # calculate vectors from the swim bladder to each eye
    eye_vectors = eye_centres - sb_c
    # assign the left and right ee based on the cross product of the vectors
    cross_product = np.cross(*eye_vectors)
    cross_sign = int(np.sign(cross_product))
    try:
        left_c, right_c = eye_centres[::cross_sign]
        left_th, right_th = eye_angles[::cross_sign]
    except ValueError:
        raise TrackingError()
    # calculate the angle in between the vectors
    inter_eye_angle = np.abs(np.degrees(cross_product / np.product(np.linalg.norm(eye_vectors, axis=1))))
    # return labelled features
    internal_features = {'sb_c': sb_c, 'sb_th': sb_th,
                         'left_c': left_c, 'left_th': left_th,
                         'right_c': right_c, 'right_th': right_th,
                         'inter_eye_angle': inter_eye_angle}
    return internal_features


def track_with_watershed(image, contours):
    """Watershed algorithm for finding the centres and angles of the eyes and swim bladder if simple thresholding fails

    The algorithm works by first fitting a triangle that encloses all the points in the internal contours of the fish.
    Using this triangle, the approximate locations of the eyes and swim bladder are calculated. These approximate
    locations are used as seeds for a watershed on the original background-subtracted image. The watershed marks
    contiguous areas of the image belonging to the same feature, from which a contour is calculated and the its centre
    and angle.

    The function is considerably slower at finding the internal features than straightforward thresholding. However, it
    is useful for when two contours fuse for a couple of frames in a recording, as occasionally happens during tracking.
    The function can still work when the fish rolls, however in these cases the eye tracking tends to be very inaccurate.
    Nonetheless, it is still useful for approximating the heading of the fish in such cases.

    Parameters
    ----------
    image : array-like
        Unsigned 8-bit integer array representing a background-subtracted image
    contours : list
        The contours that were found after applying a threshold and finding contours

    Returns
    -------
    centres, angles : np.array
        Arrays representing the centres, shape (3, 2), and angles, shape (3,), of internal features

    Raises
    ------
    TrackingError
        If any error is encountered during the watershed process. Errors tend to occur if contours is an empty list, or
        if a cv2 error is encountered when trying to calculate the minEnclosingTriangle.

    References
    ----------
    Uses slightly modified version of the watershed algorithm here:
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    """
    try:
        # find the minimum enclosing triangle for the contours
        internal_points = np.concatenate(contours, axis=0)
        ret, triangle_points = cv2.minEnclosingTriangle(internal_points)
        triangle_points = np.squeeze(triangle_points)
        # find approximate locations of the features
        triangle_centre = np.mean(triangle_points, axis=0)
        estimated_feature_centres = (triangle_points + triangle_centre) / 2
        sure_fg = np.zeros(image.shape, np.uint8)
        for c in estimated_feature_centres:
            contour_check = np.array([cv2.pointPolygonTest(cntr, array2point(c), False) for cntr in contours])
            if np.all(contour_check == -1):
                internal_points = np.squeeze(internal_points)
                distances = np.linalg.norm(internal_points - c, axis=1)
                c = internal_points[np.argmin(distances)]
            cv2.circle(sure_fg, array2point(c), 1, 255, -1)
        # watershed
        unknown = np.zeros(image.shape, np.uint8)
        cv2.drawContours(unknown, contours, -1, 255, -1)
        unknown = cv2.morphologyEx(unknown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
        unknown[sure_fg == 255] = 0
        ret, markers = cv2.connectedComponents(sure_fg, connectivity=4)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        # calculate contour features
        centres, angles = [], []
        for i in range(2, 5):
            contour_mask = np.zeros(image.shape, np.uint8)
            contour_mask[markers == i] = 255
            img, contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            c, th = contour_info(contour)
            centres.append(c)
            angles.append(th)
        centres, angles = np.array(centres), np.array(angles)
        return centres, angles
    except Exception:
        raise TrackingError()


def analyse_frame(image, background, thresh1, thresh2, n_points, track_tail=True, return_image=False):
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
    scaled = cv2.resize(bg, None, fx=2, fy=2)

    # FIND FISH
    fish_contours = find_contours(scaled, thresh1)
    mask = np.zeros(scaled.shape, np.uint8)
    masked = mask.copy()
    cv2.drawContours(mask, fish_contours, 0, 1, -1)
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
        if track_tail:
            tail_points = fit_tail(mask, sb_c, heading_vector, n_points)

        if return_image:
            contour_img = cv2.resize(image, None, fx=2, fy=2)
            tracking_img = contour_img.copy()
            colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255))
            # draw contours
            cv2.drawContours(contour_img, fish_contours, 0, colors['k'], 1)
            cv2.drawContours(contour_img, internal_contours, -1, colors['w'], 1)
            # draw tracked points
            # plot heading
            cv2.circle(tracking_img, array2point(sb_c), 3, colors['y'], -1)
            cv2.line(tracking_img, array2point(sb_c), array2point(sb_c + (80 * heading_vector)), colors['y'], 2)
            # plot eyes
            for p, v, c in zip((left_c, right_c), (left_v, right_v), ('g', 'r')):
                cv2.line(tracking_img, array2point(p - (20 * v)), array2point(p + (20 * v)), colors[c], 2)
            # plot tail points
            if track_tail:
                for p in tail_points:
                    cv2.circle(tracking_img, array2point(p), 1, colors['b'], -1)

            return contour_img, tracking_img

        tracking_params = {'centre': tuple(sb_c), 'heading': heading,
                           'left_centre': tuple(left_c), 'left_angle': left_th,
                           'right_centre': tuple(right_c), 'right_angle': right_th,
                           'tracked': True}
        if track_tail:
            tracking_params['tail_points'] = tail_points

    except TrackingError:

        if return_image:
            contour_img = cv2.resize(image, None, fx=2, fy=2)
            tracking_img = contour_img.copy()
            colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255))
            # draw contours
            cv2.drawContours(contour_img, fish_contours, 0, colors['k'], 1)
            cv2.drawContours(contour_img, internal_contours, -1, colors['w'], 1)
            return contour_img, tracking_img

        param_names = ['centre', 'heading', 'left_centre', 'left_angle', 'right_centre', 'right_angle']
        tracking_params = dict([(param_name, None) for param_name in param_names])
        if track_tail:
            tracking_params['tail_points'] = np.zeros((n_points, 2)) * np.nan
        tracking_params['tracked'] = False

    return tracking_params


def analyse_frame_tail_only(image, background, thresh1, thresh2, n_points, return_image=False):
    """Main function for tracking the tail of the fish in a video frame without tracking the eyes

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
    analyse_frame
    track_video_tail_only
    find_contours
    contour_info
    """
    # BACKGROUND SUBTRACTION
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = background_division(gray, background)
    scaled = cv2.resize(bg, None, fx=2, fy=2)

    # FIND FISH
    fish_contours = find_contours(scaled, thresh1)
    mask = np.zeros(scaled.shape, np.uint8)
    masked = mask.copy()
    cv2.drawContours(mask, fish_contours, 0, 1, -1)
    mask = mask.astype(np.bool)
    masked[mask] = scaled[mask]
    masked = cv2.equalizeHist(masked)

    # FIND HEAD
    internal_contours = find_contours(masked, thresh2)[:3]
    try:
        # find the minimum enclosing triangle for the contours
        internal_points = np.concatenate(internal_contours, axis=0)
        ret, triangle_points = cv2.minEnclosingTriangle(internal_points)
        triangle_points = np.squeeze(triangle_points)
        triangle_centre = np.mean(triangle_points, axis=0)
        distance_from_centre = np.linalg.norm(triangle_points - triangle_centre, axis=1)
        p1_idx = np.argmax(distance_from_centre)
        p1 = (triangle_points[p1_idx] + triangle_centre) / 2.
        p2 = triangle_points[np.arange(3) != p1_idx].mean(axis=0)

        heading_vector = p2 - p1
        heading_vector /= np.linalg.norm(heading_vector)
        heading = np.arctan2(*heading_vector[::-1])

        # TAIL TRACKING
        tail_points = fit_tail(mask, p1, heading_vector, n_points)

        if return_image:
            contour_img = cv2.resize(image, None, fx=2, fy=2)
            tracking_img = contour_img.copy()
            colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255))
            # draw contours
            cv2.drawContours(contour_img, fish_contours, 0, colors['k'], 1)
            cv2.drawContours(contour_img, internal_contours, -1, colors['w'], 1)
            # draw tracked points
            # plot heading
            cv2.circle(tracking_img, array2point(p1), 3, colors['y'], -1)
            cv2.line(tracking_img, array2point(p1), array2point(p1 + (80 * heading_vector)), colors['y'], 2)
            # plot tail points
            for p in tail_points:
                cv2.circle(tracking_img, array2point(p), 1, colors['b'], -1)

            return contour_img, tracking_img

        tracking_params = {'centre': tuple(p1), 'heading': heading, 'midpoint': tuple(p2),
                           'tail_points': tail_points, 'tracked': True}

    except Exception:

        if return_image:
            contour_img = cv2.resize(image, None, fx=2, fy=2)
            tracking_img = contour_img.copy()
            colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255))
            # draw contours
            cv2.drawContours(contour_img, fish_contours, 0, colors['k'], 1)
            cv2.drawContours(contour_img, internal_contours, -1, colors['w'], 1)
            return contour_img, tracking_img

        param_names = ['centre', 'heading', 'midpoint']
        tracking_params = dict([(param_name, None) for param_name in param_names])
        tracking_params['tail_points'] = np.zeros((n_points, 2)) * np.nan
        tracking_params['tracked'] = False

    return tracking_params


def track_video(video_path, background, thresh1, thresh2, n_points, track_tail=True, save_output=False,
                filename=None, output_directory=None):
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
    timer = Timer()
    timer.start()
    cap = cv2.VideoCapture(video_path)
    tracking_cols = ['centre', 'heading', 'left_centre', 'left_angle', 'right_centre', 'right_angle', 'tracked']
    tracking_dict = dict([(col, []) for col in tracking_cols])
    if track_tail:
        tail_points = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame_info = analyse_frame(frame, background, thresh1, thresh2, n_points, track_tail=track_tail)
            for col in tracking_cols:
                tracking_dict[col].append(frame_info[col])
            if track_tail:
                tail_points.append(frame_info['tail_points'])
        else:
            break
    tracking_df = pd.DataFrame(tracking_dict, columns=tracking_cols)
    if track_tail:
        tail_points = np.array(tail_points, dtype=np.float32)
    tracking_time = timer.stop()
    if save_output:
        if filename is None:
            output_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            output_name = filename
        if output_directory is None:
            tracking_path = output_name + '.csv'
            if track_tail:
                tail_points_path = output_name + '.npy'
        else:
            tracking_path = os.path.join(output_directory, output_name + '.csv')
            if track_tail:
                tail_points_path = os.path.join(output_directory, output_name + '.npy')
        tracking_df.to_csv(tracking_path, index=False)
        if track_tail:
            np.save(tail_points_path, tail_points)
        return tracking_time
    else:
        if track_tail:
            return tracking_df, tail_points
        else:
            return tracking_df


def track_video_tail_only(video_path, background, thresh1, thresh2, n_points, save_output=False,
                          filename=None, output_directory=None):
    """Main function for tracking all frames in a video without eye tracking

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
    This function either returns a DataFrame and an array (if save_output = False), or it saves the output of the
    tracking as two files (filename.csv and filename.npy) in the output_directory and returns the time it took to track
    the video (if save_output = True).

    The tracking_df contains the columns:
        'centre', 'heading', 'left_centre', 'left_angle', 'right_centre', 'right_angle', 'tracked'

    See Also
    --------
    track_video
    analyse_frame_tail_only
    """
    timer = Timer()
    timer.start()
    cap = cv2.VideoCapture(video_path)
    tracking_cols = ['centre', 'heading', 'midpoint', 'tracked']
    tracking_dict = dict([(col, []) for col in tracking_cols])
    tail_points = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame_info = analyse_frame_tail_only(frame, background, thresh1, thresh2, n_points)
            for col in tracking_cols:
                tracking_dict[col].append(frame_info[col])
            tail_points.append(frame_info['tail_points'])
        else:
            break
    tracking_df = pd.DataFrame(tracking_dict, columns=tracking_cols)
    tail_points = np.array(tail_points, dtype=np.float32)
    tracking_time = timer.stop()
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
