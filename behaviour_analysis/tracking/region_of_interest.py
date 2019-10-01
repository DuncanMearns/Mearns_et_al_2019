from ..miscellaneous import KeyboardInteraction

import cv2
import numpy as np


class RegionOfInterest(KeyboardInteraction):
    """Class for selecting rectangular ROIs in an image

    Attributes
    ----------
    image : numpy array or str
        An array representing the image or a path to an image file

    window : str
        The name of the window in the the image is displayed

    selection : bool
        Whether a rectangle has been drawn on the image

    p1 : tuple, (int, int)
        The x, y coordinates of the first point in the ROI

    p2 : tuple, (int, int)
        The x, y coordinates of the second point in the ROI
    """

    def __init__(self, image, winname='select ROI'):
        """__init__ for RegionOfInterest class

        Parameters
        ----------
        image : numpy array
            An array representing an image

        winname : str, optional (default = 'select ROI')
            The name of the window in which the image is displayed
        """
        KeyboardInteraction.__init__(self)

        if type(image) == str:
            self.image = cv2.imread(image, 0)
        else:
            self.image = image

        self.window = winname

        self.selection = False
        self.p1 = None
        self.p2 = None

    def select(self):
        """Select and ROI in the image

        Returns
        -------
        If escape key pressed - None

        If enter key pressed - ROI : tuple
            An ROI in the form ((x1, y1), (x2, y2)) where (x1, y1) are the coordinates of the top left corner of a
            rectangle and (x2, y2) are coordinates of the bottom right corner of a rectangle
        """
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._update_click)
        self._update_display()
        while True:
            self.wait(0)
            if self.enter() and self.selection:
                points = (self.p1, self.p2)
                x_min, y_min = min([p[0] for p in points]), min([p[1] for p in points])
                x_max, y_max = max([p[0] for p in points]), max([p[1] for p in points])
                ROI = (x_min, y_min), (x_max, y_max)
                cv2.destroyWindow(self.window)
                return ROI
            elif self.esc():
                print 'ROI not selected!'
                cv2.destroyWindow(self.window)
                return
            else:
                pass

    def _update_display(self):
        """Updates the image to show the selected region"""
        image = self.image.copy()
        if self.p1 and self.p2:
            cv2.rectangle(image, self.p1, self.p2, 0)
        cv2.imshow(self.window, image)

    def _update_click(self, event, x, y, flags, param):
        """Handles events passed by setMouseCallBack"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection = False
            self.p1 = (x, y)
            self.p2 = None
        elif event == cv2.EVENT_LBUTTONUP:
            self.p2 = (x, y)
            self.selection = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.selection = False
            self.p1 = None
            self.p2 = None
        elif not self.selection:
            self.p2 = (x, y)
        self._update_display()


def check_points_in_ROI(ROI, points):
    """Checks whether all of a set of points fall within a region of interest

    Parameters
    ----------
    ROI : array-like
        The top left and bottom right corners of a rectangular region of interest
    points : array-like
        An array of tail points with shape = (number_of_frames, number_of_points, 2)

    Returns
    -------
    bool
        Whether all the points in all the frames fall within the ROI
    """
    xmin, ymin = np.array(ROI[0]) * 2
    xmax, ymax = np.array(ROI[1]) * 2
    check_x = (points[:, :, 0] > xmin) & (points[:, :, 0] < xmax)
    check_y = (points[:, :, 1] > ymin) & (points[:, :, 1] < ymax)
    check_xy = np.all(check_x & check_y, axis=1)
    return np.all(check_xy)


def crop_to_rectangle(img, p1, p2):
    xmin, ymin = min(p1[0], p2[0]), min(p1[1], p2[1])
    xmax, ymax = max(p1[0], p2[0]), max(p1[1], p2[1])
    cropped = img[ymin:ymax+1, xmin:xmax+1]
    return cropped
