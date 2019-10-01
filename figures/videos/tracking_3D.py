from plotting import *
from plotting.colors import RdBuMap
from plotting.plots.animated import AnimatedPlot
from plotting.plots.kinematics import plot_tail_kinematics
from plotting.plots.tail import generate_reconstructed_points
from datasets.side_view.example_data import data
from behaviour_analysis.tracking import rotate_and_centre_image, crop_to_rectangle
from behaviour_analysis.miscellaneous import array2point
import numpy as np
from matplotlib.collections import PolyCollection
import cv2
from skimage.exposure import rescale_intensity
from matplotlib import gridspec


tinyfont = basefont.copy()
tinyfont.set_size(6)


class Tracking3DMixIn(object):

    def __init__(self, **kwargs):
        self.video = kwargs.get('video')
        self.tracking = kwargs.get('tracking')
        self.points = kwargs.get('points')
        self.kinematics = kwargs.get('kinematics')
        self.tail_columns = [col for col in self.kinematics.columns if (col[0] == 'k')]
        self.example_bouts = kwargs.get('example_bouts')
        self.top_ROI = kwargs.get('top_ROI')
        self.side_ROI = kwargs.get('side_ROI')

    def get_frame(self, i, head_stabilise=False):
        self.video.frame_change(i)
        frame = self.video.grab_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, None, fx=2, fy=2)
        frame = rescale_intensity(frame, (50, 200))
        if head_stabilise:
            centre = self.tracking.loc[i, 'side_centre'] + (2 * np.array(self.side_ROI[0]))
            pitch = self.tracking.loc[i, 'fish_elevation']
            frame = rotate_and_centre_image(frame, centre, pitch)
            h, w = frame.shape
            frame = frame[(h / 2) - 50: (h / 2) + 50, (w / 2) - 80: (w / 2) + 180]
        return frame

    def get_tracking(self, i):
        # Tracking
        fish_centre = self.tracking.loc[i, 'centre']
        right_centre, left_centre = self.tracking.loc[i, ['right_centre', 'left_centre']].apply(np.array)
        right_angle, left_angle = self.tracking.loc[i, ['right_angle', 'left_angle']]
        heading = self.tracking.loc[i, 'heading']
        right_vector = np.array([np.cos(right_angle), np.sin(right_angle)])
        left_vector = np.array([np.cos(left_angle), np.sin(left_angle)])
        heading_vector = np.array([np.cos(heading), np.sin(heading)])
        # Points
        tail_points = self.points[i]
        tail_points = np.mean(np.array([tail_points[:-1], tail_points[1:]]), axis=0)
        colors = self.kinematics.loc[i, self.tail_columns].map(np.degrees)
        colors = RdBuMap(45).map(colors)
        # All tracking info
        tracking_kwargs = dict(fish_centre=fish_centre, heading_vector=heading_vector,
                               right_centre=right_centre, left_centre=left_centre,
                               right_vector=right_vector, left_vector=left_vector,
                               tail_points=tail_points, colors=colors)
        for key in tracking_kwargs:
            if key.endswith('centre') or (key == 'tail_points'):
                tracking_kwargs[key] = np.array(tracking_kwargs[key]) + (2 * np.array(self.top_ROI[0]))
        # Side
        side_centre, hyoid, head_midpoint = self.tracking.loc[i, ['side_centre', 'hyoid', 'head_midpoint']]
        fish_elevation, head_elevation = self.tracking.loc[i, ['fish_elevation', 'head_elevation']]
        tail_vector = np.array([np.cos(fish_elevation), np.sin(fish_elevation)])
        head_vector = np.array([np.cos(head_elevation), np.sin(head_elevation)])
        side_tracking_kwargs = dict(side_centre=side_centre, hyoid=hyoid, head_midpoint=head_midpoint,
                                    tail_vector=tail_vector, head_vector=head_vector)
        for key in ['side_centre', 'hyoid', 'head_midpoint']:
            side_tracking_kwargs[key] = np.array(side_tracking_kwargs[key]) + (2 * np.array(self.side_ROI[0]))
        tracking_kwargs.update(side_tracking_kwargs)
        return tracking_kwargs


class AnimatedTracking3D(AnimatedPlot, Tracking3DMixIn):

    def __init__(self, fps=400., **kwargs):

        self.first_frame = kwargs.get('first_frame')
        self.last_frame = kwargs.get('last_frame')
        animation_frames = np.arange(self.first_frame, self.last_frame)
        AnimatedPlot.__init__(self, animation_frames, figsize=kwargs.get('figsize'))

        Tracking3DMixIn.__init__(self, **kwargs)
        self.example_bouts = kwargs.get('example_bouts')

        self.fps = fps

    def setup_figure(self):
        gs = gridspec.GridSpec(1, 2, left=0, right=1, bottom=0, top=1, wspace=0, width_ratios=(2, 3))
        self.ax1 = self.fig.add_subplot(gs[0])
        win = self.fig.canvas.window()
        win.setFixedSize(win.size())

        # ax1
        # ---

        # Show frame
        frame = self.get_frame(self.first_frame)
        self.img = self.ax1.imshow(frame, cmap='gray')
        # 300 pixels = 5 mm -> 120 pixels = 2 mm
        self.ax1.plot([100, 220], [870, 870], c='k')

        # Show tracking
        tracking_info = self.get_tracking(self.first_frame)
        # Side
        self.pitch_vector, = self.ax1.plot(*np.vstack((tracking_info['side_centre'] - (50 * tracking_info['tail_vector']),
                                                       tracking_info['side_centre'] + (150 * tracking_info['tail_vector']))).T,
                                           zorder=1, c='r', lw=0.5)
        self.elevation_vector, = self.ax1.plot(*np.vstack((tracking_info['side_centre'] - (50 * tracking_info['head_vector']),
                                                           tracking_info['side_centre'] + (50 * tracking_info['head_vector']))).T,
                                               zorder=2, c='r', lw=0.5)
        self.depression_vector, = self.ax1.plot(*np.vstack((tracking_info['head_midpoint'],
                                                            tracking_info['hyoid'])).T,
                                                zorder=2, c='c', lw=0.5)
        self.side_points = self.ax1.scatter(*np.vstack((tracking_info['side_centre'],
                                                        tracking_info['head_midpoint'])).T,
                                            c=['r', 'c'], s=3, zorder=2, lw=0)

        self.heading_vector, = self.ax1.plot(*np.vstack((tracking_info['fish_centre'],
                                                         tracking_info['fish_centre']
                                                         + (100 * tracking_info['heading_vector']))).T,
                                             c='k', lw=0.5, ls='dotted', zorder=1)

        self.tail_points = self.ax1.scatter(*tracking_info['tail_points'].T,
                                            c=tracking_info['colors'],
                                            s=1, lw=0, zorder=2)

        self.ax1.axis('off')

        # ax2
        # ---

        sub_gs = gridspec.GridSpecFromSubplotSpec(4, 1, gs[1], hspace=0.2)
        self.ax21 = self.fig.add_subplot(sub_gs[0])
        self.ax22 = self.fig.add_subplot(sub_gs[1])
        self.ax23 = self.fig.add_subplot(sub_gs[2])
        self.ax24 = self.fig.add_subplot(sub_gs[3])

        t = np.arange(self.first_frame, self.last_frame + 1) / 500.

        # Plot jaw
        self.depression, = self.ax21.plot(t, self.kinematics['depression'], c='k', lw=1)
        self.elevation, = self.ax21.plot(t, self.kinematics['elevation'].apply(np.degrees), c='0.5', lw=1)
        ymax = self.kinematics['elevation'].apply(np.degrees).max()
        self.ax21.text(t[0] + 0.02, ymax, 'Jaw depression', fontproperties=tinyfont,
                       verticalalignment='top', color='k')
        self.ax21.text(t[0] + 0.02, ymax - 3, 'Cranial elevation', fontproperties=tinyfont,
                       verticalalignment='top', color='0.5')

        # Plot tail kinematics
        self.tail_kinematics = self.kinematics.loc[:, self.tail_columns].applymap(np.degrees)
        plot_tail_kinematics(self.ax22, self.tail_kinematics, k_max=90)
        title2 = self.ax22.set_title('Rostro-caudal tail angle', loc='left', fontproperties=tinyfont)
        title2.set_position((0.005, 0.62))

        # Plot tail angle
        ps = generate_reconstructed_points(self.kinematics.loc[:, self.tail_columns].values, 0)
        self.tip_angle_data = np.degrees(np.arcsin(ps[:, 1, -1] / 50.))
        self.tip_angle, = self.ax23.plot(t, self.tip_angle_data, c='k', lw=1)
        self.ax23.set_ylim(-45, 20)
        self.ax23.text(t[0] + 0.02, self.tip_angle_data.max() - 5, 'Tail tip angle', fontproperties=tinyfont,
                       verticalalignment='bottom', color='k')

        # Plot pitch
        self.pitch, = self.ax24.plot(t, self.kinematics['fish_elevation'].apply(np.degrees), c='k', lw=1)
        self.ax24.text(t[0] + 0.02, self.kinematics['fish_elevation'].apply(np.degrees).max(),
                       'Pitch', fontproperties=tinyfont, verticalalignment='top', color='k')

        # Inset
        # -----
        fig_width, fig_height = self.fig.get_size_inches()
        ratio = float(fig_width) / fig_height
        x0, y0, x1, y1 = self.ax1.get_position().bounds
        w = 0.4 * (x1 - x0)
        y = (y1 - y0) / 2.
        h = ratio * w / 2.6
        self.ax3 = self.fig.add_axes((x1 - (1.2 * w), y - (h / 2.), w, h))
        frame = self.get_frame(self.first_frame, head_stabilise=True)
        self.inset = self.ax3.imshow(frame, cmap='gray')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])

        # Set axis limits
        for ax in (self.ax21, self.ax23, self.ax24):
            ax.set_xlim(t[0], t[-1])
            ax.axis('off')

    def animate(self, i):
        self.update_ax1(i)
        self.update_ax2(i)
        self.update_ax3(i)

    def update_ax1(self, i):
        # Update image
        frame = self.get_frame(i)
        self.img.set_data(frame)
        # Update tracking
        tracking_info = self.get_tracking(i)

        pitch_vector = np.vstack((tracking_info['side_centre'] - (50 * tracking_info['tail_vector']),
                                  tracking_info['side_centre'] + (150 * tracking_info['tail_vector']))).T
        elevation_vector = np.vstack((tracking_info['side_centre'] - (50 * tracking_info['head_vector']),
                                      tracking_info['side_centre'] + (50 * tracking_info['head_vector']))).T
        depression_vector = np.vstack((tracking_info['head_midpoint'], tracking_info['hyoid'])).T
        self.pitch_vector.set_data(*pitch_vector)
        self.elevation_vector.set_data(*elevation_vector)
        self.depression_vector.set_data(*depression_vector)
        self.side_points.remove()
        self.side_points = self.ax1.scatter(*np.vstack((tracking_info['side_centre'],
                                                        tracking_info['head_midpoint'])).T,
                                            c=['r', 'c'], s=3, zorder=2, lw=0)

        heading_vector = np.vstack((tracking_info['fish_centre'],
                                    tracking_info['fish_centre'] + (80 * tracking_info['heading_vector']))).T
        self.heading_vector.set_data(*heading_vector)
        self.tail_points.remove()
        self.tail_points = self.ax1.scatter(*tracking_info['tail_points'].T,
                                            c=tracking_info['colors'],
                                            s=1, lw=0, zorder=2)

    def update_ax2(self, i):
        t = np.arange(self.first_frame, i + 1) / 500.
        # Update jaw
        self.depression.set_data(t, self.kinematics['depression'].values[:1 + i - self.first_frame])
        self.elevation.set_data(t, np.degrees(self.kinematics['elevation'].values[:1 + i - self.first_frame]))
        # Update til kinematics
        tail_kinematics = np.zeros(self.tail_kinematics.shape)
        tail_kinematics[0:1 + i - self.first_frame] = self.tail_kinematics.loc[self.first_frame:i].values
        self.ax22.get_images()[0].set_data(tail_kinematics.T)
        # Update tail tip
        self.tip_angle.set_data(t, self.tip_angle_data[:1 + i - self.first_frame])
        for child in self.ax23.get_children():
            if isinstance(child, PolyCollection):
                child.remove()
                del child
        for idx, bout_info in self.example_bouts.iterrows():
            if i > bout_info.start:
                t_ = np.arange(bout_info.start, min(i, bout_info.end) + 1) / 500.
                self.ax23.fill_between(t_, np.ones((len(t_),)) * (-45), np.ones((len(t_),)) * 20, facecolor='0.8')
        # Update pitch
        self.pitch.set_data(t, np.degrees(self.kinematics['fish_elevation'].values[:1 + i - self.first_frame]))

    def update_ax3(self, i):
        frame = self.get_frame(i, head_stabilise=True)
        self.inset.set_data(frame)


if __name__ == "__main__":

    tracking_plot = AnimatedTracking3D(fps=40., figsize=(6, 2), **data)
    tracking_plot.setup_figure()
    # tracking_plot.show_frame(data['last_frame'])
    # tracking_plot.play()
    tracking_plot.save(os.path.join(output_directory, 'video7'))
