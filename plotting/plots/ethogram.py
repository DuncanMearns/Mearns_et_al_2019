from plotting.colors import cluster_colors
import numpy as np
from matplotlib.patches import Polygon


order = [5, 1, 6, 0, 3, 2, 4]  # ccw from top


def plot_ethogram(ax, P, show, alpha_values=None, arrow_color=None):
    if isinstance(show, bool):
        show = (np.ones((7, 7), 'i4') * show).astype('bool')

    thetas = np.linspace(0, 2 * np.pi, 8)[:-1] + np.pi / 2
    nodes = np.array([np.cos(thetas), np.sin(thetas)]).T

    ax.scatter(*nodes.T, c=cluster_colors[order])

    for n in range(7):
        node_start = nodes[n]
        j = order[n]
        for i, p in enumerate(P[:, j]):
            if p > 0.01:
                width = max(0.01, p * 0.2)
                if show[i, j]:
                    if alpha_values is not None:
                        alpha = alpha_values[i, j]
                    else:
                        alpha = 1
                    color = cluster_colors[j]
                    if arrow_color is not None:
                        color = arrow_color
                else:
                    alpha = 1
                    color = '0.7'
                if (i == j):
                    r1 = 0.1
                    r2 = r1 + (width / 2)
                    th = np.linspace(np.radians(-50), np.radians(230), 50)
                    p1 = r1 * np.array([np.cos(th), np.sin(th)]).T
                    p2 = r2 * np.array([np.cos(th), np.sin(th)]).T
                    path = np.concatenate([p1, p2[::-1]])
                    rotation = np.arctan2(*node_start[::-1]) - (np.pi / 2)
                    R = np.array([[np.cos(rotation), np.sin(rotation)],
                                  [-np.sin(rotation), np.cos(rotation)]])
                    path = np.dot(path, R)
                    path = path + (node_start * 1.2)
                    arrow = Polygon(path, linewidth=0, color=color, alpha=alpha)
                    ax.add_patch(arrow)
                else:
                    node_end = nodes[order.index(i)]
                    v = (node_end - node_start)
                    v_ = np.array([v[1], -v[0]])
                    length = np.linalg.norm(v)
                    dx, dy = v * (length - 0.3) / length
                    x, y = (0.005 * v_) + node_start + (0.15 * v / length)
                    head_width = width * 3
                    head_length = min(head_width * 1.5, 0.3)
                    ax.arrow(x, y, dx, dy,
                             width=width,
                             length_includes_head=True,
                             head_width=head_width,
                             head_length=head_length,
                             color=color,
                             alpha=alpha,
                             linewidth=0,
                             shape='left')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')


def plot_ethogram_difference(ax, P, show, alpha_values=None):
    if isinstance(show, bool):
        show = (np.ones((7, 7), 'i4') * show).astype('bool')

    thetas = np.linspace(0, 2 * np.pi, 8)[:-1] + np.pi / 2
    nodes = np.array([np.cos(thetas), np.sin(thetas)]).T

    ax.scatter(*nodes.T, c=cluster_colors[order])

    for n in range(7):
        node_start = nodes[n]
        j = order[n]
        for i, p in enumerate(P[:, j]):
            if p > 0:
                width = min(0.1, p * 0.1)
                if show[i, j]:
                    alpha = 1
                    color = cluster_colors[j]
                    if (i == j):
                        r1 = 0.1
                        r2 = r1 + (width / 2)
                        th = np.linspace(np.radians(-50), np.radians(230), 50)
                        p1 = r1 * np.array([np.cos(th), np.sin(th)]).T
                        p2 = r2 * np.array([np.cos(th), np.sin(th)]).T
                        path = np.concatenate([p1, p2[::-1]])
                        rotation = np.arctan2(*node_start[::-1]) - (np.pi / 2)
                        R = np.array([[np.cos(rotation), np.sin(rotation)],
                                      [-np.sin(rotation), np.cos(rotation)]])
                        path = np.dot(path, R)
                        path = path + (node_start * 1.2)
                        arrow = Polygon(path, linewidth=0, color=color, alpha=alpha)
                        ax.add_patch(arrow)
                    else:
                        node_end = nodes[order.index(i)]
                        v = (node_end - node_start)
                        v_ = np.array([v[1], -v[0]])
                        length = np.linalg.norm(v)
                        dx, dy = v * (length - 0.3) / length
                        x, y = (0.005 * v_) + node_start + (0.15 * v / length)
                        head_width = width * 3
                        head_length = min(head_width * 1.5, 0.3)
                        ax.arrow(x, y, dx, dy,
                                 width=width,
                                 length_includes_head=True,
                                 head_width=head_width,
                                 head_length=head_length,
                                 color=color,
                                 alpha=alpha,
                                 linewidth=0,
                                 shape='left')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
