import numpy as np
import colorsys
from datasets.main_dataset import experiment
import os


isomap = np.load(os.path.join(experiment.subdirs['analysis'], 'isomap.npy'))[:, :3]


class IsomapColors():

    def __init__(self):

        angle = (np.arctan2(isomap[:, 1], isomap[:, 0]))
        z = isomap[:, 2]
        self.colors = self.map_colors(angle, z)

        self.ecolors = self.colors * 0.8
        self.ecolors[:, -1] = 1

    @staticmethod
    def map_colors(angle, z):
        if angle.ndim > 1:
            angle = angle[:-1, 1:].flatten()
        h = angle / (2 * np.pi)
        h = (h - 0.5) % 1

        zmin, zmax = isomap[:, 2].min(), isomap[:, 2].max()
        z_ = (z - zmin) / float(zmax - zmin)
        if z.ndim == 1:
            l = 1 - z_
        else:
            l = 1 - (z_[:-1, 1:]).flatten()

        s = np.ones((len(h),))
        hls = zip(h, l, s)

        colors = np.array([colorsys.hls_to_rgb(*c) for c in hls])
        colors = np.concatenate([colors, np.ones((len(colors), 1)) * 0.8], axis=1)

        return colors

    def get_color(self, i):
        return self.colors[i]

    def get_ecolor(self, i):
        return self.ecolors[i]


isomap_colors = IsomapColors()
