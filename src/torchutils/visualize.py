import numpy as np
from itertools import product

from torch import nn

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.axis import XTick, YTick


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
        elif isinstance(tick, XTick):
            tick.label1.set_verticalalignment('middle')


def add_sublabels(axis, ticks, ticklabels, level):
    pass
    #TODO

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
        if level > len(dims):
            return
            
        ticklabels = list(reversed(list(zip(*product(*[range(tensor.size(dim)) for dim in dims])))))[level]
        ticklabels = list(map(str, ticklabels))
        n_ticks = len(ticklabels) + 1
        ticks = np.arange(n_ticks) - 0.5
        if level==0:
            center_ticks(axis, ticks, ticklabels)
        else:
            add_sublabels(axis, ticks, ticklabels, level, previous_label=dims[-level])
        
        if level == len(dims):
            axis.set_label(dims[0])


        
    for axis, dims in zip([ax.xaxis, ax.yaxis], [xdims, ydims]):
        add_labels(axis, dims)

    plt.colorbar()
    return image
