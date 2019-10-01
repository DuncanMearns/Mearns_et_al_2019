import numpy as np


def plot_tail_kinematics(ax, k, fs=500., **kwargs):
    # Make sure variables are correct type
    k = np.array(k)
    fs = float(fs)
    # Unpack kwargs
    bout_length = len(k) / fs
    t_lim = kwargs.get('t_lim', (0, bout_length))
    first_frame = int(t_lim[0] * fs)
    last_frame = int(t_lim[1] * fs)
    k_max = kwargs.get('k_max', np.abs(k).max())
    ax_lim = kwargs.get('ax_lim', t_lim)
    # Select data to display
    k_display = k[first_frame:last_frame]
    # Plot data
    ax.imshow(k_display.T, aspect='auto', extent=(t_lim[0], t_lim[1], 0, 1), origin='lower', cmap='RdBu_r',
              clim=(-k_max, k_max), interpolation='bilinear')
    # Adjust axes
    ax.set_xlim(ax_lim)
    ax.set_ylim(1, 0)
    ax.axis('off')
