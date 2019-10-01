from ..miscellaneous import KeyboardInteraction, read_csv, array2point
from ..video import Video
from .tracking import analyse_frame, analyse_frame_tail_only

import cv2
import numpy as np
import sys
from ast import literal_eval


def rotate_and_centre_image(image, centre, angle, fill_outside=255):
    """Rotates and centres an image

    Parameters
    ----------
    image : array-like
        An image represented as an array

    centre : tuple, list or array-like
        The point in the image that is to become the centre (x, y) coordinates

    angle : float
        The angle through which to rotate the image (radians)

    fill_outside : int, optional (0-255)
        The grayscale value to fill points outside the original image

    Returns
    -------
    stabilised : array
        The image centred on the given point and rotated by the given angle
    """
    height, width = image.shape
    x_shift = (width / 2.) - centre[0]
    y_shift = (height / 2.) - centre[1]
    M = np.array([[1, 0, x_shift], [0, 1, y_shift]])
    centred = cv2.warpAffine(image, M, (width, height), borderValue=fill_outside)
    R = cv2.getRotationMatrix2D((width / 2, height / 2), np.degrees(angle), 1)
    stabilised = cv2.warpAffine(centred, R, (width, height), borderValue=fill_outside)
    return stabilised


def show_tracking(frame, centre, heading, left_centre, left_angle, right_centre, right_angle, points, tracked):
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
    colors = dict(k=0, w=(255, 255, 255), b=(255, 0, 0), g=(0, 255, 0), r=(0, 0, 255), y=(0, 255, 255))
    # draw tracked points
    if tracked:
        # plot heading
        heading_vector = np.array([np.cos(heading), np.sin(heading)])
        cv2.circle(img, array2point(centre), 3, colors['y'], -1)
        cv2.line(img, array2point(centre), array2point(centre + (80 * heading_vector)), colors['y'], 2)
        # plot eyes
        left_v = np.array([np.cos(left_angle), np.sin(left_angle)])
        right_v = np.array([np.cos(right_angle), np.sin(right_angle)])
        for p, v, c in zip((left_centre, right_centre), (left_v, right_v), ('g', 'r')):
            cv2.line(img, array2point(p - (20 * v)), array2point(p + (20 * v)), colors[c], 2)
        # plot tail points
        for p in points:
            cv2.circle(img, array2point(p), 1, colors['b'], -1)
    return img


def check_tracking(video, tracking, points, **kwargs):
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
    k : int
        ASCII code for a pressed key
    """
    # Get the video, tracking and points objects
    if type(video) == str:
        v = Video(video)
    else:
        v = video
    if type(tracking) == str:
        tracking_df = read_csv(tracking, centre=literal_eval, left_centre=literal_eval, right_centre=literal_eval)
    else:
        tracking_df = tracking
    if type(points) == str:
        tail_points = np.load(points)
    else:
        tail_points = points
    # Check tracking
    key_input = KeyboardInteraction()
    if 'winname' in kwargs.keys():
        winname = kwargs['winname']
    else:
        winname = 'check tracking'
    cv2.namedWindow(winname)
    cv2.createTrackbar('frame', winname, 0, v.frame_count - 1, v.frame_change)
    while True:
        frame = v.grab_frame()
        tracking_kwargs = tracking_df.loc[v.frame_number]
        image = show_tracking(frame, points=tail_points[v.frame_number], **tracking_kwargs)
        cv2.imshow(winname, image)
        k = cv2.waitKey(1)
        if key_input.valid(k):
            cv2.destroyWindow(winname)
            break
    return k


def set_thresholds(video_paths, background_path, thresh1_initial=10, thresh2_initial=210, n_points=51, track_eyes=True):
    """Set the thresholds for finding the fish and internal contours in videos

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

    track_eyes : bool, optional (default = True)
        Whether the eyes should be tracked

    Returns
    -------
    thresh1, thresh2 : int
        Values for thresholds to use for tracking

    Notes
    -----
    Creates windows to set thresholds for tracking.
    The 'set thresholds' window contains track bars for adjusting the frame number and threshold values.
    The 'contours' window displays contours that were found using the current thresholds.
    The 'tracking' window displays the final output of the tracking obtained using the current thresholds.

    First, thresh1 should be adjusted so that the fish is properly outlined in the 'contours' window.
    Then, thresh2 should be adjusted so that the eyes and swim bladder are outlined in the majority of frames.

    Pressing the space key opens the next video.
    Pressing the enter key returns the current thresholds.
    Pressing the escape key quits using sys.exit().

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
    background = cv2.imread(background_path, 0)
    thresh1, thresh2 = thresh1_initial, thresh2_initial
    key_input = KeyboardInteraction()
    for video_path in video_paths:
        v = Video(video_path)
        # setup the windows
        cv2.namedWindow('contours')
        cv2.namedWindow('tracking')
        for window in ['contours', 'tracking']:
            cv2.createTrackbar('frame', window, 0, v.frame_count - 1, v.frame_change)
        cv2.createTrackbar('thresh1', 'contours', thresh1, 255, lambda x: x)
        cv2.createTrackbar('thresh2', 'contours', thresh2, 255, lambda x: x)
        while True:
            for window in ['contours', 'tracking']:
                cv2.setTrackbarPos('frame', window, v.frame_number)
            thresh1 = cv2.getTrackbarPos('thresh1', 'contours')
            thresh2 = cv2.getTrackbarPos('thresh2', 'contours')

            frame = v.grab_frame()
            if track_eyes:
                contours, tracking = analyse_frame(frame, background, thresh1, thresh2, n_points, return_image=True)
            else:
                contours, tracking = analyse_frame_tail_only(frame, background, thresh1, thresh2, n_points, return_image=True)
            cv2.imshow('contours', contours)
            cv2.imshow('tracking', tracking)

            key_input.wait(1)
            if key_input.valid():
                break

        if key_input.enter():
            cv2.destroyAllWindows()
            break
        elif key_input.esc():
            cv2.destroyAllWindows()
            sys.exit()
        else:
            pass

    return thresh1, thresh2
