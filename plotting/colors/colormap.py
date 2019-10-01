from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
import numpy as np


class ColorMap(object):

    def __init__(self, cmap, clim):
        self.colormap = cm.ScalarMappable(cmap=cmap, norm=Normalize(*clim))

    def map(self, *args):
        return np.array([self.colormap.to_rgba(a) for a in args]).squeeze()


class RdBuMap(ColorMap):

    def __init__(self, cmax=90):
        ColorMap.__init__(self, cmap='RdBu_r', clim=(-cmax, cmax))


class TimeMap(ColorMap):

    def __init__(self, cmin, cmax):
        ColorMap.__init__(self, cmap='viridis', clim=(cmin, cmax))
