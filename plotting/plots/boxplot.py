from .. import open_ax
from matplotlib import pyplot as plt


def boxplot(group_values, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    bp = ax.boxplot(group_values, patch_artist=True,flierprops=dict(marker='.', markersize=3, markeredgewidth=0.5))
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
        plt.setp(bp[element], color='k', linewidth=0.5)
    plt.setp(bp['medians'], color='k', linewidth=1)

    group_colors = kwargs.get('group_colors', ['w'] * len(group_values))
    for i, patch in enumerate(bp['boxes']):
        patch.set(facecolor=group_colors[i])

    open_ax(ax)
    return ax
