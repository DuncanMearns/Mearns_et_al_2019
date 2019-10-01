from ..miscellaneous import KeyboardInteraction
import cv2
import os


class Video(KeyboardInteraction):
    """Simple class for handling videos using OpenCV

    This class is helpful when interacting with videos using openCV and displaying them in a cv2 window (see examples)

    Attributes
    ----------
    path : str
        The path to the video file

    cap : cv2.VideoCapture
        The openCV VideoCapture object

    frame_count : int
        Number of frames in the video

    frame_number : int
        Number of the current frame
    """

    def __init__(self, video_path):
        """__init__ function for Video class

        Parameters
        ----------
        video_path : str
            Complete path to a video file (.avi)
        """
        KeyboardInteraction.__init__(self)
        self.path = video_path
        self.cap = cv2.VideoCapture(self.path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0

    def frame_change(self, val):
        """Updates the frame number

        Parameters
        ----------
        val : int
            New value for the frame number
        """
        self.frame_number = val

    def grab_frame(self):
        """Grabs the current frame as defined by the frame number

        Returns
        -------
        frame : np.array
            The current frame as an unsigned 8-bit integer array
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            raise Exception('frame does not exist!')

    def scroll(self, **kwargs):
        first_frame = kwargs.get('first_frame', 0)
        last_frame = kwargs.get('last_frame', self.frame_count)
        name = kwargs.get('name', os.path.basename(self.path))
        n_frames = last_frame - first_frame
        cv2.namedWindow(name)
        cv2.createTrackbar('frame', name, 0, n_frames, lambda x: x)
        self.frame_change(first_frame)
        while True:
            frame_number = cv2.getTrackbarPos('frame', name) + first_frame
            self.frame_change(frame_number)
            frame = self.grab_frame()
            cv2.imshow(name, frame)
            self.wait(1)
            if self.valid():
                break
        cv2.destroyWindow(name)
        return self.k

    def play(self, **kwargs):
        first_frame = kwargs.get('first_frame', 0)
        last_frame = kwargs.get('last_frame', self.frame_count)
        name = kwargs.get('name', os.path.basename(self.path))
        frame_rate = kwargs.get('frame_rate', self.cap.get(cv2.CAP_PROP_FPS))
        frame_time = max(1, int(1000. / frame_rate))
        cv2.namedWindow(name)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        for f in range(first_frame, last_frame):
            self.frame_change(f)
            ret, frame = self.cap.read()
            cv2.imshow(name, frame)
            self.wait(frame_time)
            if self.valid():
                break
        cv2.destroyWindow(name)
        return self.k

    def return_frames(self, first_frame, last_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        frames = []
        for f in range(first_frame, last_frame + 1):
            self.frame_change(f)
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            else:
                raise Exception('frame does not exist!')
        return frames
