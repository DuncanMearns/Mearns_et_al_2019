import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import is_color_like
from ..colors import *


def trajectory_to_bout(trajectory, eigenfish, mean, std):
    return (np.dot(trajectory, eigenfish) * std) + mean


def generate_reconstructed_points(k, rotation):
    R = np.array([[np.cos(np.radians(rotation)), -np.sin(np.radians(rotation))],
                  [np.sin(np.radians(rotation)), np.cos(np.radians(rotation))]])
    all_points = []
    for i, tail_shape in enumerate(k):
        vs = np.array([-np.cos(tail_shape), np.sin(tail_shape)]).T
        ps = np.zeros((len(vs) + 1, 2))
        ps[1:] = np.cumsum(vs, axis=0)
        if rotation:
            ps = np.dot(R, ps.T)
        else:
            ps = ps.T
        all_points.append(ps)
    return np.array(all_points)


def plot_reconstructed_points(ax, points,  fs=500., color='time', lw=3, **kwargs):
    """Plot an overlay of tail points on a given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes instance on which to plot the tail points.
    points : np.ndarray
        An array of xy coordinates representing the tail in each frame with shape (n_frames, 2, n_points).
    fs : float
        The sampling frequency at which data were acquired.
    color : str or Colormap or list-like, optional
        If string, should be either a valid matplotlib color or {'t', 'time'} or 'binary'. If 't' or 'time' the
        standard TimeMap is used. If 'binary' the Binary colormap is used. Otherwise, a Colormap instance for computing
        the color of the tail in each frame. Otherwise, a list of colors corresponding to each frame.
    lw : int or float, optional
        The line width of the tail.

    Other Parameters
    ----------------
    t_lim : tuple of floats
        The first and last time frame to plot (in seconds). Default behavior is to plot all frames.
    alpha : float
        Alpha value (between 0 and 1).
    c_lim : tuple
        Tuple of floats. Upper and lower values for initializing the TimeMap. Default is the same as t_lim.

    Returns
    -------
    None
    """
    fs = float(fs)
    plotting_kwargs = dict(lw=lw)
    # Calculate frames to plot
    bout_length = len(points) / fs
    t_lim = kwargs.get('t_lim', (0, bout_length))
    first_frame = int(t_lim[0] * fs)
    last_frame = int(t_lim[1] * fs)
    ps = points[first_frame:last_frame]
    # Set colors
    if is_color_like(color):
        colors = color
        plotting_kwargs['alpha'] = kwargs.get('alpha', 1)
    elif isinstance(color, str):
        if color.lower() in ('t', 'time'):  # If standard TimeMap being used
            c_lim = kwargs.get('c_lim', t_lim)
            colors = TimeMap(*c_lim).map(np.linspace(t_lim[0], t_lim[1], len(ps)))
        elif color == 'binary':  # If BinaryMap being used
            colors = ColorMap('binary', (0, len(ps) - 1)).map(np.arange(len(ps)))
        else:
            raise ValueError('c must be valid color')
        plotting_kwargs['alpha'] = kwargs.get('alpha', 1)
    elif isinstance(color, ColorMap):  # If an alternative Colormap is being used
        colors = color.map(np.arange(len(ps)))
    else:  # If a list of colors is being used
        colors = color[-len(ps):]
    # Plot
    for i, (xs, ys) in enumerate(ps):
        ax.plot(xs, ys, color=colors[i], **plotting_kwargs)


def plot_trajectory(ax, t, fs=500., color='t', lw=3, fill=False, **kwargs):
    """Plot a trajectory in principal component space on a given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or matplotlib.axes.Axes3D
        The Axes instance on which to plot the trajectory.
    t : np.ndarray
        A 2 dimensional array of points defining the trajectory with shape (n_points, n_dimensions).
    fs : float
        The sampling frequency at which data were acquired.
    color : str or Colormap or list-like, optional
        If string, should be either a valid matplotlib color or {'t', 'time'}, in which case the standard TimeMap is
        used. Otherwise, a Colormap instance for computing the color of the trajectory segments. Otherwise, a list of
        colors corresponding to each segment of the trajectory.
    lw : int or float, optional
        The line width of the trajectory.
    fill : bool, optional
        Whether to fill in gaps between line segments. Create a smoother trajectory for thicker line widths but may
        produce unexpected or unwanted behavior where lines cross.

    Other Parameters
    ----------------
    projection : str, {'2d', '3d'}
        Whether to plot a 2- or 3-dimensional trajectory.
    t_lim : tuple of floats
        The first and last time point of the trajectory to plot (in seconds).
        Default behavior is to plot the complete trajectory.
    alpha : float
        Alpha value (between 0 and 1).
    c_lim : tuple
        Tuple of floats. Upper and lower values for initializing the TimeMap. Default is the same as t_lim.
    x_lim, y_lim, z_lim : tuple
        Max and min values of the x, y, and z axes. Default is plus/minus absolute maximum along each axis.

    Returns
    -------
    None
    """
    # Check projection type
    if (kwargs.get('projection', '2d') == '3d') or ax.name == '3d':
        projection_3d = True
    else:
        projection_3d = False
    # Make sure variables are correct type and shape
    if projection_3d:
        t = np.array(t[:, :3])
    else:
        t = np.array(t[:, :2])
    fs = float(fs)
    # Calculate portion of trajectory to plot
    bout_length = len(t) / fs
    t_lim = kwargs.get('t_lim', (0, bout_length))
    first_frame = int(t_lim[0] * fs)
    last_frame = int(t_lim[1] * fs)
    t_display = t[first_frame:last_frame]
    # Get the color scheme
    if is_color_like(color):
        # Plot using plt.plot
        color = color
        alpha = kwargs.get('alpha', 1)
        ax.plot(*t_display.T, c=color, lw=lw, alpha=alpha)
    else:
        # Plot as line segments
        ps = np.expand_dims(t_display, 1)
        segments = np.concatenate([ps[:-1], ps[1:]], axis=1)
        # Set colors
        if isinstance(color, str):
            if color.lower() in ('t', 'time'):  # If standard TimeMap being used
                c_lim = kwargs.get('c_lim', t_lim)
                colors = TimeMap(*c_lim).map(np.linspace(t_lim[0], t_lim[1], len(segments)))
            else:
                raise ValueError('c must be valid color')
        elif isinstance(color, ColorMap):  # If an alternative Colormap is being used
            colors = color.map(np.linspace(t_lim[0], t_lim[1], len(segments)))
        else:  # If a list of colors is being used
            colors = color[-len(segments):]
        # Plot line segments
        if projection_3d:
            lc = Line3DCollection(segments, colors=colors, lw=lw)
        else:
            lc = LineCollection(segments, colors=colors, lw=lw)
        ax.add_collection(lc)
        # Fill in gaps in trajectory
        if fill:
            if len(colors) > 0:
                colors = list(colors)
                colors.append(colors[-1])
                colors = np.array(colors)[:len(t_display)]
                if kwargs.get('alpha', None) is not None:
                    alpha = kwargs.get('alpha')
                    colors = colors[:, :3]
                    ax.scatter(*t_display.T, lw=0, s=lw ** 2, c=colors, alpha=alpha)
                else:
                    colors[:, -1] = np.max([np.zeros((len(colors),)), colors[:, -1] - 0.2], axis=0)
                    for i in range(len(t_display)):
                        ax.scatter(*t_display[[i]].T, lw=0, s=lw**2, c=colors[i, :3], alpha=colors[i, -1])
    # Adjust axes
    ax_lims = np.ceil(np.abs(t).max(axis=0))
    x_lim = kwargs.get('x_lim', (-ax_lims[0], ax_lims[0]))
    ax.set_xlim(*x_lim)
    y_lim = kwargs.get('y_lim', (-ax_lims[1], ax_lims[1]))
    ax.set_ylim(*y_lim)
    if projection_3d:
        z_lim = kwargs.get('z_lim', (-ax_lims[2], ax_lims[2]))
        ax.set_zlim(z_lim)
