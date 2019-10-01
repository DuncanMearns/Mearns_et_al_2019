from plotting import *
from plotting.plots.animated import AnimatedPlot
from plotting.plots.kinematics import plot_tail_kinematics
from plotting.plots.tail import generate_reconstructed_points
from plotting.colors.colormap import RdBuMap
from datasets.main_dataset.example_data import data
import numpy as np
from matplotlib import gridspec
from behaviour_analysis.tracking import rotate_and_centre_image
from matplotlib.collections import PolyCollection
import cv2
from skimage.exposure import rescale_intensity


class TrackingMixIn(object):

    def __init__(self, **kwargs):
        self.video = kwargs.get('video')
        self.tracking = kwargs.get('tracking')
        self.points = kwargs.get('points')
        self.kinematics = kwargs.get('kinematics')
        self.tail_columns = [col for col in self.kinematics.columns if (col[0] == 'k')]
        self.example_bouts = kwargs.get('example_bouts')

    def get_frame(self, i, head_stabilise=False):
        self.video.frame_change(i)
        frame = self.video.grab_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, None, fx=2, fy=2)
        frame = rescale_intensity(frame, (50, 200))[:1000, :1000]
        if head_stabilise:
            frame = rotate_and_centre_image(frame, *self.tracking.loc[i, ['centre', 'heading']])
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
        return tracking_kwargs


class AnimatedTracking(AnimatedPlot, TrackingMixIn):

    def __init__(self, fps=500., **kwargs):

        self.first_frame = kwargs.get('first_frame')
        self.last_frame = kwargs.get('last_frame')
        animation_frames = np.arange(self.first_frame, self.last_frame)
        AnimatedPlot.__init__(self, animation_frames, figsize=kwargs.get('figsize'))

        TrackingMixIn.__init__(self, **kwargs)
        self.example_bouts = kwargs.get('example_bouts')

        self.fps = fps

    def setup_figure(self):

        win = self.fig.canvas.window()
        win.setFixedSize(win.size())

        gs = gridspec.GridSpec(1, 2, left=0, right=1, bottom=0, top=0.92, wspace=0.06, width_ratios=(1, 2))
        self.ax1 = self.fig.add_subplot(gs[0])

        # ax1
        # ---

        # Show frame
        frame = self.get_frame(self.first_frame)
        self.img = self.ax1.imshow(frame, cmap='gray')
        # Scale bar
        h, w = frame.shape
        w -= 50
        self.ax1.plot([100, 100 + (w / 5)], [h - 80, h - 80], c='k', lw=2)

        # Show tracking
        tracking_info = self.get_tracking(self.first_frame)
        self.heading_vector, = self.ax1.plot(*np.vstack((tracking_info['fish_centre'],
                                                         tracking_info['fish_centre']
                                                         + (80 * tracking_info['heading_vector']))).T,
                                             c='k', lw=1, ls='dotted', zorder=1)
        self.eye_centres = self.ax1.scatter(*np.vstack((tracking_info['right_centre'],
                                                         tracking_info['left_centre'])).T,
                                            c=['m', 'g'], s=5, lw=0)
        self.eye_vectors = []
        for p, v, c in ((tracking_info['right_centre'], tracking_info['right_vector'], 'm'),
                        (tracking_info['left_centre'], tracking_info['left_vector'], 'g')):
            eye_vector, = self.ax1.plot(*np.vstack((p - (30 * v), p + (30 * v))).T, c=c, lw=1)
            self.eye_vectors.append(eye_vector)

        self.tail_points = self.ax1.scatter(*tracking_info['tail_points'].T,
                                            c=tracking_info['colors'],
                                            s=2, lw=0, zorder=2)

        self.ax1.axis('off')

        # ax2
        # ---

        sub_gs = gridspec.GridSpecFromSubplotSpec(3, 1, gs[1], wspace=0, hspace=0.1)

        self.ax21_r = self.fig.add_subplot(sub_gs[0])
        self.ax22 = self.fig.add_subplot(sub_gs[1])
        self.ax23 = self.fig.add_subplot(sub_gs[2])

        t = np.arange(self.first_frame, self.last_frame + 1) / 500.

        # ---------
        # Plot eyes
        # ---------
        self.right_eye_data = self.kinematics['right'].apply(np.degrees)
        self.left_eye_data = self.kinematics['left'].apply(np.degrees)
        eye_max = np.ceil(max(self.right_eye_data.max(), (-self.left_eye_data).max()))
        eye_min = np.floor(min(self.right_eye_data.min(), (-self.left_eye_data).min()))
        eye_range = eye_max - eye_min

        self.right_eye, = self.ax21_r.plot(t, self.right_eye_data, c='m', lw=1)
        self.ax21_r.set_ylim(eye_min, eye_min + (2 * eye_range))

        self.ax21_l = plt.twinx(self.ax21_r)
        self.left_eye, = self.ax21_l.plot(t, self.left_eye_data, c='g', lw=1)
        self.ax21_l.set_ylim(-eye_min - (2 * eye_range), -eye_min)

        self.ax21_l.text(t[0] + 0.02, self.left_eye_data.iloc[0] - 5, 'L',
                         fontproperties=verysmallfont, verticalalignment='top', color='g')
        self.ax21_r.text(t[0] + 0.02, self.right_eye_data.iloc[0], 'R',
                         fontproperties=verysmallfont, verticalalignment='bottom', color='m')

        self.ax21_l.spines['left'].set_bounds(-40, -10)
        self.ax21_l.set_yticks([-40, -10])
        self.ax21_l.set_yticklabels([40, 10], fontproperties=verysmallfont)
        self.ax21_l.set_yticks(np.arange(-30, -10, 10), minor=True)

        self.ax21_r.spines['left'].set_bounds(10, 40)
        self.ax21_r.set_yticks([10, 40])
        self.ax21_r.set_yticklabels([10, ], fontproperties=verysmallfont)
        self.ax21_r.set_yticks(np.arange(20, 40, 10), minor=True)

        title1 = self.ax21_r.set_title(u'Eye angle (\u00b0)', loc='left', fontproperties=verysmallfont)
        title1.set_position((0.005, 0.9))

        # Axis limits
        for ax in (self.ax21_r, self.ax21_l):
            ax.set_xlim(t[0], t[-1])
            open_ax(ax)
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_color('0.5')
            ax.spines['left'].set_linewidth(1)
            ax.tick_params(axis='y', which='both', color='0.5', length=3, width=1, labelcolor='0.5', pad=1)
            ax.tick_params(axis='y', which='minor', length=2, width=1)

        # --------------------
        # Plot tail kinematics
        # --------------------
        self.tail_kinematics = self.kinematics.loc[:, self.tail_columns].applymap(np.degrees)
        plot_tail_kinematics(self.ax22, self.tail_kinematics, k_max=90)
        title2 = self.ax22.set_title('Rostro-caudal tail angle', loc='left', fontproperties=verysmallfont)
        title2.set_position((0.005, 0.65))

        # Create color bar
        (l, b, w, h) = self.ax22.get_position().bounds
        cbar = self.fig.add_axes((l + (w * 0.4), b + (h * 0.8), w * 0.05, h * 0.1))
        cm = plt.cm.ScalarMappable(cmap='RdBu')
        cm.set_array(np.linspace(-1, 1, 2))
        cb = self.fig.colorbar(cm, cax=cbar, orientation='horizontal', ticks=[])
        cb.outline.set_linewidth(1)
        cbar.text(-0.1, 0, u'90\u00b0(L)', fontproperties=verysmallfont, horizontalalignment='right')
        cbar.text(1.1, 0, u'90\u00b0(R)', fontproperties=verysmallfont, horizontalalignment='left')

        # Axis label
        self.ax22.text(-0.03, 0.05, 'Ro', fontproperties=verysmallfont, horizontalalignment='right', verticalalignment='top')
        self.ax22.text(-0.03, 1, 'Ca', fontproperties=verysmallfont, horizontalalignment='right', verticalalignment='bottom')

        # ---------------
        # Plot tail angle
        # ---------------
        ps = generate_reconstructed_points(self.kinematics.loc[:, self.tail_columns].values, 0)
        self.tip_angle_data = np.degrees(np.arcsin(ps[:, 1, -1] / 50.))

        self.tip_angle, = self.ax23.plot(t, self.tip_angle_data, c='k', lw=0.5)

        # Axis limits
        open_ax(self.ax23)
        self.ax23.set_xlim(t[0], t[-1])
        self.ax23.set_xticks([])
        self.ax23.spines['bottom'].set_visible(False)

        self.ax23.set_ylim(-50, 30)
        self.ax23.plot([t[-20] - 0.5, t[-20]], [-45, -45], c='k', lw=2)
        self.ax23.set_yticks([-30, 0, 30])
        self.ax23.spines['left'].set_bounds(-30, 30)
        self.ax23.spines['left'].set_color('0.5')
        self.ax23.spines['left'].set_linewidth(1)
        self.ax23.tick_params(axis='y', which='both', color='0.5', length=3, width=1, labelcolor='0.5', pad=1)
        self.ax23.set_yticklabels([-30, 0, 30], fontproperties=verysmallfont)
        self.ax23.tick_params(axis='y', which='minor', length=2, width=1)
        self.ax23.set_yticks(np.arange(-20, 30, 10), minor=True)

        title3 = self.ax23.set_title(u'Tail tip angle (\u00b0)', loc='left', fontproperties=verysmallfont)
        title3.set_position((0.005, 0.78))

    def animate(self, i):
        self.update_ax1(i)
        self.update_ax2(i)

    def update_ax1(self, i):
        # Update image
        frame = self.get_frame(i)
        self.img.set_data(frame)
        # Update tracking
        tracking_info = self.get_tracking(i)
        heading_vector = np.vstack((tracking_info['fish_centre'],
                                    tracking_info['fish_centre'] + (80 * tracking_info['heading_vector']))).T
        eye_centres = np.vstack((tracking_info['right_centre'], tracking_info['left_centre'])).T
        self.heading_vector.set_data(*heading_vector)
        self.eye_centres.remove()
        self.eye_centres = self.ax1.scatter(* eye_centres,
                                            c=['m', 'g'], s=5, lw=0)
        for i, which in enumerate(['right', 'left']):
            p = tracking_info[which + '_centre']
            v = tracking_info[which + '_vector']
            eye_vector = np.vstack((p - (30 * v), p + (30 * v))).T
            self.eye_vectors[i].set_data(*eye_vector)
        self.tail_points.remove()
        self.tail_points = self.ax1.scatter(*tracking_info['tail_points'].T,
                                            c=tracking_info['colors'],
                                            s=2, lw=0, zorder=2)

    def update_ax2(self, i):
        t = np.arange(self.first_frame, i + 1) / 500.
        # Update eyes
        self.right_eye.set_data(t, self.right_eye_data.loc[self.first_frame:i])
        self.left_eye.set_data(t, self.left_eye_data.loc[self.first_frame:i])
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
                self.ax23.fill_between(t_, np.ones((len(t_),)) * (-40), np.ones((len(t_),)) * 20, facecolor='0.8')


if __name__ == "__main__":

    tracking_plot = AnimatedTracking(fps=50., figsize=(6, 2), **data)
    tracking_plot.setup_figure()
    # tracking_plot.show_frame(data['last_frame'])
    # tracking_plot.play()
    tracking_plot.save(os.path.join(output_directory, 'video1'))
