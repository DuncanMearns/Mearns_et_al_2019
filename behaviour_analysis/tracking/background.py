from ..miscellaneous import Timer
import cv2
import numpy as np


def calculate_projection(*video_paths, **kwargs):
    """Calculates a projection for one or multiple videos

    Parameters
    ----------
    *video_paths : str (if one) or list of strings (if multiple)
        List of strings representing paths to video files
    **kwargs : dict, optional
        sample_factor : int, default 500
            Down-sampling rate for videos when calculating the projection
        projection_type : str, default 'mean'
            The type of projection to perform. Accepted values:
                'mean', 'average' : calculates a mean intensity projection
                'median' : calculates a median intensity projection
                'min', 'minimum' : calculates a minimum intensity projection
                'max', 'maximum' : calculates a maximum intensity projection

    Returns
    -------
    projection : numpy array
        A projection (defined by the projection_type) of all the videos down-sampled by the sample_factor
    """
    if 'sample_factor' in kwargs.keys():
        sample_factor = kwargs['sample_factor']
    else:
        sample_factor = 500
    all_frames = []
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        sample_frames = np.arange(0, cap.get(cv2.CAP_PROP_FRAME_COUNT), sample_factor)
        for framenumber in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenumber)
            ret, frame = cap.read()
            if ret:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_frames.append(frame)
        cap.release()
    all_frames = np.array(all_frames)
    if 'projection_type' in kwargs.keys():
        if kwargs['projection_type'] in ['mean', 'average']:
            projection = np.mean(all_frames, axis=0)
        elif kwargs['projection_type'] == 'median':
            projection = np.median(all_frames, axis=0)
        elif kwargs['projection_type'] in ['min', 'minimum']:
            projection = np.min(all_frames, axis=0)
        elif kwargs['projection_type'] in ['max', 'maximum']:
            projection = np.max(all_frames, axis=0)
        else:
            print 'unknown projection type, calculating mean projection'
            projection = np.mean(all_frames, axis=0)
    else:
        projection = np.mean(all_frames, axis=0)
    return projection


def background_division(image, background):
    """Subtracts the background from an image then divides by the background and rescales the result

    This has advantages over simple background subtraction as it allows for easier thresholding if the background does
    not have uniform illumination

    Parameters
    ----------
    image : array like
        The image to perform background division on
    background : array like
        The background image

    Returns
    -------
    clipped : array like
        The result of the background division as an unsigned 8-bit array

    See Also
    --------
    background_subtraction
    """
    img = image.astype('float64')
    bg = background.astype('float64')
    div = (bg - img) / (bg + 1)
    div *= 255
    clipped = np.clip(div, 0, 255).astype('uint8')
    return clipped


def background_subtraction(image, background):
    """Simple background subtraction - assumes foreground objects are darker than the background

    If background illumination is not uniform and a thresholding step is going be used, divide_background may have
    better performance

    Parameters
    ----------
    image : array like
        The image to perform background subtraction on
    background : array like
        The background image

    Returns
    -------
    new : array like
        The result of the background subtraction as an unsigned 8-bit array

    See Also
    --------
    background_division
    """
    bg = background.astype('i4')
    new = bg - image
    new = np.clip(new, 0, 255)
    new = new.astype('uint8')
    return new


def save_background(output_path, *video_paths, **kwargs):
    timer = Timer()
    timer.start()
    background = calculate_projection(*video_paths, **kwargs)
    background = background.astype('uint8')
    cv2.imwrite(output_path, background)
    time_taken = timer.stop()
    return time_taken
