import numpy as np
from itertools import product

from torch import nn

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.axis import XAxis, XTick, YTick


def center_ticks(axis, ticks, ticklabels):
    ticks = np.asarray(ticks)
    assert len(ticks) == len(ticklabels) + 1
    assert ticks.ndim == 1

    axis.set_major_locator(ticker.FixedLocator(ticks))
    axis.set_minor_locator(ticker.FixedLocator((ticks[1:] + ticks[:-1]) / 2))

    axis.set_major_formatter(ticker.NullFormatter())
    axis.set_minor_formatter(ticker.FixedFormatter(ticklabels))

    for tick in axis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        if isinstance(tick, XTick):
            tick.label1.set_horizontalalignment('center')
        elif isinstance(tick, YTick):
            tick.label1.set_verticalalignment('center')


def add_sublabels(axis, ticks, ticklabels, level, previous_label="", loc=None):
    assert len(ticks) - 1 == len(ticklabels)
    if loc is None:
        loc = "bottom" if isinstance(axis, XAxis) else "left"
    
    # Add sublabel lines
    ax = axis.axes
    sgn = 1 if loc in ["right", "top"] else -1
    offset = 0.12 * sgn * level + (1 if loc in ["right", "top"] else 0)
    if isinstance(axis, XAxis):
        trans = ax.get_xaxis_transform()
        get_xy = lambda pos: (pos, offset)
        ha = "center"
        va = "top" if loc=="bottom" else "bottom"
        lim = ax.get_xlim()
        dx = (max(ax.get_xlim()) - min(ax.get_xlim())) / 100
        get_plot_xy = lambda start, stop, offset: ([start + dx, stop - dx], [offset, offset])
        rotation = "horizontal"
    else:
        trans = ax.get_yaxis_transform()
        get_xy = lambda pos: (offset, pos)
        ha = "right" if loc=="left" else "left"
        va = "center"
        lim = ax.get_ylim()
        dy = (max(ax.get_ylim()) - min(ax.get_ylim())) / 100
        get_plot_xy = lambda start, stop, offset: ([offset, offset], [start + dy, stop - dy])
        rotation = "vertical"

    def ra(a):
        if a == "left":
            return "right"
        if a == "right":
            return "left"
        if a == "bottom":
            return "top"
        if a == "top":
            return "bottom"
        return a
    
    for start, stop, ticklabel in zip(ticks[:-1], ticks[1:], ticklabels):
        pos = (start + stop) / 2
        ax.annotate(ticklabel, xy=get_xy(pos), xycoords=trans, ha=ha, va=va, rotation=rotation)
        ax.annotate(previous_label, xy=get_xy(pos), xycoords=trans, ha=ra(ha), va=ra(va), rotation=rotation)
        ax.plot(*get_plot_xy(start, stop, offset), color="grey", transform=trans, clip_on=False)
    

def tensorshow(tensor, xdims, ydims, fignum=None):
    assert hasattr(tensor, "names")

    # Handle overlap between xdims and ydims
    xydims =  set(xdims).intersection(set(ydims))
    xdims = [dim + ("x" if dim in xydims else "") for dim in xdims]
    ydims = [dim + ("y" if dim in xydims else "") for dim in ydims]

    for dim in xydims:
        s = tensor.size(dim).sqrt().ceil().int()
        tensor = nn.ConstantPad1d((0, s**2 - tensor.size(dim)), float("nan"))(tensor)
        tensor = tensor.unflatten(dim, ((dim + "x", s), (dim + "y", s)))

    # Tensor -> Matrix
    tensor = tensor.align_to(*xdims, *ydims)
    matrix = tensor.flatten(xdims, "x").flatten(ydims, "y")

    # Heatmap
    image = plt.matshow(matrix, fignum=fignum)
    ax = image.axes

    # Add hierarchical axes labels
    def add_labels(axis, dims, level=0):
        if level == len(dims):
            if isinstance(axis, XAxis):
                ax.set_xlabel(dims[0])
            else:
                ax.set_ylabel(dims[0])
        if level >= len(dims):
            return
            
        ticklabels = list(reversed(list(zip(*product(*[range(tensor.size(dim)) for dim in dims])))))[level]
        ticklabels = np.asarray(list(map(str, ticklabels)))
        n_ticks = len(ticklabels) + 1
        ticks = np.arange(n_ticks) - 0.5
        if level==0:
            center_ticks(axis, ticks, ticklabels)
        else:
            mask = np.hstack([[True], ticklabels[:-1]!=ticklabels[1:]])
            mticks = np.hstack([ticks[:-1][mask], ticks[-1]])
            mticklabels = np.array(ticklabels)[mask]
            add_sublabels(
                axis,
                mticks,
                mticklabels,
                level,
                previous_label=dims[-level],
                loc="top" if isinstance(axis, XAxis) else "left",
            )
        
        return add_labels(axis, dims, level=level + 1)

        
    for axis, dims in zip([ax.xaxis, ax.yaxis], [xdims, ydims]):
        add_labels(axis, dims)

    plt.colorbar()
    return image