import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # Register 3d projection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


DEFAULT_DPI = 100.0
BASIC_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
HIGHTLIGHT_COLORS = ['red', 'blue', 'yellow', 'black', 'green']


def getColor(groups, use_max=True):
    """
    Inputs:
        groups: (N, G)
    Returns:
        a list of colors
    """
    if use_max:
        groups = np.argmax(groups, -1)
        return [BASIC_COLORS[int(groups[i])] for i in range(groups.shape[0])]
    else:
        g_min = np.amin(groups)
        groups -= g_min
        raise NotImplementedError


def toy_render(point_cloud,
               shape=None,
               title=None,
               highlight_idxs=[],
               groups=None,
               gt_groups=None,
               xlim=(-0.7, 0.7),
               ylim=(-0.7, 0.7),
               zlim=(0, 0.7)):
    """
    Inputs:
        point_cloud: (T, N, 3)
        groups: (T, N, G)
        gt_groups: (N,)
    Returns:
        a list of frames [img1, img2, ...]
    """
    if shape is not None:
        figsize = (shape[0] / DEFAULT_DPI,
                   shape[1] / DEFAULT_DPI)
    else:
        figsize = None

    frames = []
    for i_f in range(point_cloud.shape[0]):
        xs = point_cloud[i_f, :, 0]
        zs = point_cloud[i_f, :, 1]
        ys = point_cloud[i_f, :, 2]

        if gt_groups is not None:
            # gt_groups (N,)
            color = [BASIC_COLORS[gt_groups[i]]
                     for i in range(gt_groups.shape[0])]
        elif groups is not None:
            # groups (T, N, G)
            color = getColor(groups[i_f].squeeze())
        else:
            # groups is None and gt_groups is None:
            color = ['gray' for _ in range(point_cloud.shape[1])]

        for j, hl_id in enumerate(highlight_idxs):
            color[hl_id] = HIGHTLIGHT_COLORS[j % len(HIGHTLIGHT_COLORS)]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        if title is not None:
            ax.set_title(title)

        fig.canvas.draw()

        frame = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        plt.close(fig)

    return np.stack(frames)


def plot_curves(x, ys,
                save_path=None,
                curve_labels=[],
                x_label='',
                y_label='',
                title=''):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, y in enumerate(ys):
        if i < len(curve_labels):
            ax.plot(x, y, label=curve_labels[i])
        else:
            ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
