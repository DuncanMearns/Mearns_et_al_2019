from behaviour_analysis.tracking import find_contours
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np


def smoothed_histogram(histogram, average, threshold=50, sigma=1.5):
    fish = find_contours(average.astype('uint8'), threshold)
    masked = np.zeros(average.shape, dtype='uint8')
    cv2.drawContours(masked, fish, 0, 1, -1)
    masked = masked.astype('bool')
    histogram[masked] = 0
    histogram /= histogram.sum()
    histogram = gaussian_filter(histogram, sigma)
    return histogram
