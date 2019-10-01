from plotting import *
from plotting.plots.animated import AnimatedPlot
from plotting.plots.tail import generate_reconstructed_points, plot_trajectory, plot_reconstructed_points
from plotting.colors import TimeMap
from datasets.main_dataset.example_data import experiment, data
from tracking_2D import TrackingMixIn
from behaviour_analysis.analysis.bouts import whiten_data, map_data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import numpy as np


class AnimatedTransformed(AnimatedPlot, TrackingMixIn):

    def __init__(self, animation_frames, fps=500, **kwargs):
        AnimatedPlot.__init__(self, animation_frames, figsize=kwargs.get('figsize', (4, 3)))
        TrackingMixIn.__init__(self, **kwargs)
        self.fps = fps

    def setup_figure(self):
        gs = gridspec.GridSpec(2, 3, left=0, right=0.98, bottom=0.02, top=0.98, wspace=0.1, hspace=0)

        self.ax1 = self.fig.add_subplot(gs[:, :2], projection='3d')
        self.ax2 = self.fig.add_subplot(gs[0, 2])
        self.ax3 = self.fig.add_subplot(gs[1, 2])

        self.xmax, self.ymax, self.zmax = np.ceil(np.abs(transformed).max(axis=0))
        self.xticks, self.yticks, self.zticks = [0], [0], [0]
        self.xticklabels, self.yticklabels, self.zticklabels = [], [], []

        frame = self.get_frame(first_frame, head_stabilise=True)

        self.img = self.ax2.imshow(frame, cmap='gray')
        self.ax2.set_xlim(250, 600)
        self.ax2.set_ylim(675, 325)
        self.ax2.axis('off')
        self.ax2.set_title('Head-stabilized frame', loc='center', fontproperties=verysmallfont,
                           verticalalignment='center')


    def animate(self, i):

        # Update ax1
        start = max(0, i - first_frame - 100)
        end = 1 + i - first_frame
        self.ax1.clear()
        self.ax1.set_title('Tail shape represented in\n3D coordinate space defined\nby principal components',
                           loc='center', fontproperties=verysmallfont, verticalalignment='center')
        plot_trajectory(self.ax1, transformed[start:end],
                        fill=True, color=colors, lw=2,
                        x_lim=(-self.xmax, self.xmax),
                        y_lim=(-self.ymax, self.ymax),
                        z_lim=(-self.zmax, self.zmax))
        self.ax1.set_xticks(self.xticks)
        self.ax1.set_yticks(self.yticks)
        self.ax1.set_zticks(self.zticks)
        self.ax1.set_xticklabels(self.xticklabels)
        self.ax1.set_yticklabels(self.yticklabels)
        self.ax1.set_zticklabels(self.zticklabels)
        self.ax1.set_xlabel('PC1', labelpad=-10)
        self.ax1.set_ylabel('PC2', labelpad=-10)
        self.ax1.set_zlabel('PC3', labelpad=-10)
        self.ax1.view_init(15, 120)

        # Update ax2
        frame = self.get_frame(i, head_stabilise=True)
        self.img.set_data(frame)

        # Update ax3
        self.ax3.clear()
        title = self.ax3.set_title('Tail reconstruction\nfrom principal components',
                                   loc='center', fontproperties=verysmallfont)
        title.set_position((0.5, 0.8))
        plot_reconstructed_points(self.ax3, reconstructed[start:end], color=colors, lw=1)
        self.ax3.set_xlim(-60, 30)
        self.ax3.set_ylim(-45, 45)
        self.ax3.axis('off')


if __name__ == "__main__":

    eigenfish_path = os.path.join(experiment.directory, 'analysis', 'behaviour_space', 'eigenfish.npy')
    tail_statistics_path = os.path.join(experiment.directory, 'analysis', 'behaviour_space', 'tail_statistics.npy')
    eigenfish = np.load(eigenfish_path)
    mean, std = np.load(tail_statistics_path)

    kinematics = data['kinematics']
    first_frame, last_frame = data['first_frame'], data['last_frame']
    tail_columns = [col for col in kinematics.columns if (col[0] == 'k')]

    tail = kinematics.loc[first_frame:last_frame, tail_columns].values
    whitened = whiten_data(tail, mean, std)
    transformed = map_data(whitened, eigenfish[:3])
    reconstructed = generate_reconstructed_points(np.dot(transformed, eigenfish[:3]) * std + mean, 0)

    colors = TimeMap(0, 0.2).map(np.linspace(-0.2, 0.2, 101))
    colors = np.array(colors)
    colors[:50, -1] = np.linspace(0, 1, 50)

    transformed_plot = AnimatedTransformed(range(first_frame, last_frame), fps=50, figsize=(4, 3), **data)
    transformed_plot.setup_figure()
    # transformed_plot.show_frame(last_frame)
    # transformed_plot.play()
    transformed_plot.save(os.path.join(output_directory, 'video2_mpg'))
