from matplotlib import pyplot as plt
from matplotlib import animation
import os


class AnimatedPlot(object):

    def __init__(self, animation_frames, **kwargs):

        self.fig = plt.figure(dpi=300, **kwargs)
        self.animation_frames = animation_frames
        self.fps = 100

    def setup_figure(self):
        pass

    def animate(self, i):
        pass

    def play(self):
        ani = animation.FuncAnimation(self.fig, self.animate, self.animation_frames,
                                      interval=int(1000 / self.fps),
                                      blit=False)
        plt.show()

    def _save_as_ffmpeg(self, fname):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=4000)
        ani = animation.FuncAnimation(self.fig, self.animate, self.animation_frames,
                                      interval=int(1000 / self.fps),
                                      blit=False)
        ani.save(fname + '.mp4', writer=writer)

    def _save_as_tif(self, fname):
        if not os.path.exists(fname):
            os.makedirs(fname)
        for i in self.animation_frames:
            self.animate(i)
            self.fig.savefig(os.path.join(fname, str(i) + '.tiff'), dpi=300)

    def save(self, fname, as_tif=False):
        if as_tif:
            self._save_as_tif(fname)
        else:
            self._save_as_ffmpeg(fname)

    def show_frame(self, i):
        self.animate(i)
        self.show()

    @ staticmethod
    def show():
        plt.show()
