from .. import open_ax
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def violinplot(group_values, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    values = np.concatenate(group_values)
    labels = np.concatenate([[i] * len(group) for (i, group) in enumerate(group_values)])
    df = pd.DataFrame(dict(value=values, label=labels))

    group_colors = kwargs.get('group_colors', ['w'] * len(group_values))

    sns.violinplot(x='label', y='value', data=df, ax=ax, color=group_colors)
    sns.despine(ax=ax)

    open_ax(ax)
    return ax
