from plotting import *
from matplotlib import pyplot as plt
from matplotlib.colorbar import ColorbarBase


width = 1
height = 5

for cmap in ('PiYG', 'Blues', 'Greens', 'Reds', 'YlOrRd',
             'inferno', 'magma', 'viridis', 'bwr', 'plasma',
             'bone', 'winter', 'autumn', 'binary'):

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))

    cb = ColorbarBase(ax, cmap=cmap)
    ax.axis('off')

    save_fig(fig, 'colorbars', cmap)
# plt.show()
