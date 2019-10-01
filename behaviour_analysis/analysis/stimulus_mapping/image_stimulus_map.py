from ...tracking import rotate_and_centre_image
import cv2
import numpy as np


def image_stimulus_map(img, tracking_info):
    """Generates a map of the visual scene in fish-centred coordinates.

    Parameters
    ----------
    img : np.array
        Frame from video
    tracking_info : pd.Series or dict
        Tracking data for given frame

    Returns
    -------
    cropped : np.array
        Head-centred thresholded cropped image
    """
    h, w = img.shape
    blur = 255 - cv2.GaussianBlur(img.astype('uint8'), (3, 3), 0)
    threshed = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
    centred = rotate_and_centre_image(255 - threshed, np.array(tracking_info['midpoint']) / 2., tracking_info['heading'], 0)
    cropped = centred[h / 4: 3 * h / 4, w / 4: 3 * w / 4]
    return cropped
