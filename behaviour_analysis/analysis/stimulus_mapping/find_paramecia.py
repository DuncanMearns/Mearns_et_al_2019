from behaviour_analysis.tracking import contour_info
import cv2
import numpy as np
from scipy import ndimage


def find_paramecia(img):
    """Finds putative paramecia within a stimulus map"""
    filt = ndimage.median_filter(img, 5)
    threshed = ndimage.grey_erosion(filt, 3) > 5
    threshed = threshed.astype('uint8') * 255
    im2, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres, orientations = zip(*[contour_info(cntr) for cntr in contours if cv2.contourArea(cntr) > 3])
    centres = np.array(centres)
    orientations = np.array(orientations)
    return centres, orientations
