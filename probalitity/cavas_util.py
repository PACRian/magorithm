import numpy as np
from matplotlib import pyplot as plt


def draw_path(x, y, ax=None, **kwargs):
    x_pth = np.tile(x, (2, 1)).ravel(order='F')[1:]
    y_pth = np.tile(y, (2, 1)).ravel(order='F')[:-1]
    if ax is None:
        return plt.plot(x_pth, y_pth, **kwargs)
    else:
        return ax.plot(x_pth, y_pth, **kwargs)

def set_minor_ticks(ax, x_perc=4, y_perc=4):
    minor_xticks = np.vstack([np.convolve(ax.get_xticks()[1:-1], [k, x_perc-k], 'valid') \
        for k in range(1, x_perc)]).ravel(order='F')/x_perc
    minor_yticks = np.vstack([np.convolve(ax.get_yticks()[1:-1], [k, y_perc-k], 'valid') \
        for k in range(1, y_perc)]).ravel(order='F')/y_perc
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)