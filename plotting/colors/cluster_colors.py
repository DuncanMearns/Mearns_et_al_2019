from .colormap import ColorMap
import numpy as np

cmap = ColorMap('Set1', (0, 8))
cluster_colors = np.array(cmap.map(np.array([0, 7, 2, 4, 1, 3, 6])))
