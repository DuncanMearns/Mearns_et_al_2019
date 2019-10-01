from plotting.plots import AnimatedPlot
from matplotlib import gridspec
import numpy as np


class StimulusSequencePlot(AnimatedPlot):

    def __init__(self, histogram1, histogram2, average1, average2, height=3.0, **kwargs):

        assert len(histogram1) == len(histogram2)

        self.histogram1 = histogram1
        self.histogram2 = histogram2

        self.average1 = average1
        self.average2 = average2

        self.label1 = kwargs.get('label1', dict(name='label1', c='w'))
        self.label2 = kwargs.get('label2', dict(name='label2', c='w'))

        self.height, self.width = self.histogram1.shape[1:]
        wh_ratio = self.width / float(self.height)

        histmax = np.max([self.histogram1, self.histogram2])
        self.levels = np.linspace(0.0005, histmax, 10)

        super(StimulusSequencePlot, self).__init__(len(histogram1), figsize=(2 * height * wh_ratio, height))
        self.fps = 25

    def setup_figure(self):

        gs = gridspec.GridSpec(1, 2, left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)

        # ===========
        # Create axes
        # ===========

        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        for ax in (self.ax1, self.ax2):
            ax.axis('off')

        # ================
        # Draw static text
        # ================

        self.ax1.text(self.width / 2, 8, self.label1['name'], dict(color=self.label1['c'], fontsize=8, ha='center'))
        self.ax2.text(self.width / 2, 8, self.label2['name'], dict(color=self.label2['c'], fontsize=8, ha='center'))

        # ================
        # Plot first frame
        # ================

        self.im1 = self.ax1.imshow(self.average1[0].T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
        self.contour1 = self.ax1.contourf(self.histogram1[0], cmap='magma', levels=self.levels)

        self.im2 = self.ax2.imshow(self.average2[0].T[::-1], cmap='binary_r', vmin=0, vmax=255, interpolation='bilinear')
        self.contour2 = self.ax2.contourf(self.histogram2[0], cmap='magma', levels=self.levels)

        self.text = self.ax2.text(2 * self.width / 3, 17, self.time(0),
                                  dict(color=(1, 1, 1, 0.75), fontsize=6, ha='right'))

        self.ax1.plot([2, 17.15], [96, 96], c='w', lw=1.5)

        # ===============
        # Create colorbar
        # ===============

        (l, b, w, h) = self.ax1.get_position().bounds

        self.cbar = self.fig.add_axes((w * 0.05, h * 0.7, w * 0.05, h * 0.15))
        cb = self.fig.colorbar(self.contour1, cax=self.cbar, ticks=self.levels[[0, -1]], orientation='vertical')
        cb.outline.set_linewidth(0)

        histmax = self.float_to_string(self.levels[-1])
        self.cbar.set_yticks([0, self.levels[-1]])
        self.cbar.set_yticklabels([0, histmax], dict(color=(1, 1, 1, 0.75), fontsize=4, ha='left'))
        self.cbar.tick_params(axis='y', pad=1, length=0)

        self.cbar.set_ylabel('Prob.\ndensity', dict(color=(1, 1, 1, 0.75), fontsize=4, rotation=0, ha='left'))
        self.cbar.yaxis.set_label_coords(1.5, 0.75)

    def animate(self, i):
        self.im1.set_array(self.average1[i].T[::-1])
        for c in self.contour1.collections:
            c.remove()
        self.contour1 = self.ax1.contourf(self.histogram1[i], cmap='magma', levels=self.levels)

        self.im2.set_array(self.average2[i].T[::-1])
        for c in self.contour2.collections:
            c.remove()
        self.contour2 = self.ax2.contourf(self.histogram2[i], cmap='magma', levels=self.levels)

        self.text.set_text(self.time(i))

    @staticmethod
    def time(t):
        return str(int(1000 * (t - 450) / 500.)) + ' ms'

    @staticmethod
    def float_to_string(f):
        s = str(float(f))
        pre, post = s.split('.')
        non_zero = [(lambda i: not (int(i) == 0))(char) for char in post]
        n = non_zero.index(True)
        rounded = np.round(f, n + 1)
        return str(rounded)