from matplotlib import pyplot as plt
from isomap_plot import isomap_plot


def transition_mode_plot(Vs, Va, **kwargs):

    fig, axes = plt.subplots(2, 2, figsize=(3, 3))

    kwargs['cmap'] = 'bwr'
    kwargs['vmin'] = -0.08
    kwargs['vmax'] = 0.08

    isomap_plot(axes[0][0], c=Vs[:, 0], **kwargs)
    isomap_plot(axes[0][1], c=Vs[:, 1], **kwargs)
    isomap_plot(axes[1][0], c=Va[:, 0], **kwargs)
    isomap_plot(axes[1][1], c=Va[:, 1], **kwargs)

    axes[1][1].set_yticks([])
    axes[1][1].set_yticks([], minor=True)
    axes[1][1].spines['left'].set_visible(False)

    return fig
