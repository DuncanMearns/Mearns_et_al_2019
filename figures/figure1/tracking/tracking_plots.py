from plotting import *
from plotting.colors.colormap import RdBuMap
from datasets.main_dataset.example_data import data
from datasets.schematics import prey_capture_assay as schematic
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
from matplotlib.patches import Rectangle


def plot_schematic_frame(schematic, frame, rect, scale):
    """
    Fig: schematic + example frame
     ___  ______
    |   ||      |
    |   ||      |
    |___||______|
    """

    box_l, box_r, box_t, box_b = rect
    assert frame.shape[0] == frame.shape[1]
    w2, h2 = frame.shape[1], frame.shape[0]
    h1 = h2
    w1 = h1 * schematic.shape[1] / float(schematic.shape[0])
    wpad = 0.05 * w2

    width = w1 + wpad + w2
    height = h1

    fig_width = scale * width
    fig_height = scale * height

    ax1_box = (0, 0, w1 / width, 1)
    ax2_box = ((w1 + wpad) / width, 0, w2 / width, 1)

    # Setup figures
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax1 = fig.add_axes(ax1_box)
    ax2 = fig.add_axes(ax2_box)

    # Show images
    ax1.imshow(schematic, interpolation='bilinear')
    ax2.imshow(frame, interpolation='bilinear')

    # Show bounding box
    ax2.add_patch(Rectangle((box_l, box_t), box_r - box_l, box_b - box_t, fill=False, ec='k', lw=1, ls='dashed'))

    # Turn axes off
    for ax in (ax1, ax2):
        ax.axis('off')

    return fig


def plot_tracked_fish(frame, rect, scale, lw=3, s=10, **tracking_kwargs):
    """
    Fig: example tracking
     _______
    |       |
    |_______|
    """

    box_l, box_r, box_t, box_b = rect

    width = (box_r - box_l)
    height = (box_b - box_t)

    fig_width = scale * width
    fig_height = scale * height

    # Setup figures
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes((0, 0, 1, 1))

    # Show image
    ax.imshow(frame, interpolation='bilinear')

    # Set x and y limits
    ax.set_xlim(box_l, box_r)
    ax.set_ylim(box_b, box_t)

    # Plot tracking
    fish_centre = tracking_kwargs['fish_centre']
    heading_vector = tracking_kwargs['heading_vector']
    right_centre = tracking_kwargs['right_centre']
    left_centre = tracking_kwargs['left_centre']
    right_vector = tracking_kwargs['right_vector']
    left_vector = tracking_kwargs['left_vector']
    tail_points = tracking_kwargs['tail_points']
    colors = tracking_kwargs['colors']

    ax.plot(*np.vstack((fish_centre - (160 * heading_vector), fish_centre + (80 * heading_vector))).T, c='k', lw=lw,
             ls='dotted', zorder=1)
    ax.scatter(*np.vstack((right_centre, left_centre)).T, c=['m', 'g'], s=3*s)
    for p, v, c in ((right_centre, right_vector, 'm'), (left_centre, left_vector, 'g')):
        ax.plot(*np.vstack((p - (30 * v), p + (30 * v))).T, c=c, lw=lw)
    ax.scatter(*tail_points.T, c=colors, s=s, zorder=2)

    # Turn axes off
    ax.axis('off')

    return fig


if __name__ == "__main__":

    video = data['video']
    tracking = data['tracking']
    kinematics = data['kinematics']
    points = data['points']
    example_frame = data['example_frame']

    # Get tracking data
    fish_centre = tracking.loc[example_frame, 'centre']
    right_centre, left_centre = tracking.loc[example_frame, ['right_centre', 'left_centre']].apply(np.array)
    right_angle, left_angle = tracking.loc[example_frame, ['right_angle', 'left_angle']]
    heading = tracking.loc[example_frame, 'heading']
    right_vector = np.array([np.cos(right_angle), np.sin(right_angle)])
    left_vector = np.array([np.cos(left_angle), np.sin(left_angle)])
    heading_vector = np.array([np.cos(heading), np.sin(heading)])

    # Grab frame
    video.frame_change(example_frame)
    frame = video.grab_frame()
    frame = cv2.resize(frame, None, fx=2, fy=2)
    frame = rescale_intensity(frame, (50, 200))[:1000, :1000]

    # Get kinematic data
    tail_columns = [col for col in kinematics.columns if col[0] == 'k']

    # Get tail points
    tail_points = points[example_frame]
    tail_points = np.mean(np.array([tail_points[:-1], tail_points[1:]]), axis=0)
    colors = kinematics.loc[example_frame, tail_columns].map(np.degrees)
    colors = RdBuMap(45).map(colors)

    # ====
    # PLOT
    # ====

    box_l = fish_centre[0] - 180
    box_r = fish_centre[0] + 80
    box_t = fish_centre[1] - 60
    box_b = fish_centre[1] + 60

    fig1 = plot_schematic_frame(schematic, frame, (box_l, box_r, box_t, box_b), 1)

    tracking_kwargs = dict(fish_centre=fish_centre, heading_vector=heading_vector,
                           right_centre=right_centre, left_centre=left_centre,
                           right_vector=right_vector, left_vector=left_vector,
                           tail_points=tail_points, colors=colors)
    fig2 = plot_tracked_fish(frame, (box_l, box_r, box_t, box_b), scale=2.5, lw=2, s=5, **tracking_kwargs)

    print 'Fig1 width:height ratio =', fig1.get_figwidth() / fig1.get_figheight()

    final_width = 3.15
    scale_factor = final_width / fig1.get_figwidth()
    for fig in (fig1, fig2):
        fig.set_size_inches(fig.get_figwidth() * scale_factor, fig.get_figheight() * scale_factor)
    print fig1.get_figheight()

    # plt.show()
    save_fig(fig1, 'figure1', 'schematic+frame_2D')
    save_fig(fig2, 'figure1', 'tracked_frame_2D')
