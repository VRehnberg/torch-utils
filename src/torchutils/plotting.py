from torch import nn
import matplotlib.pyplot as plt


def tensorshow(tensor, xdims, ydims, fignum=None):
    assert hasattr(tensor, "names")
    xydims =  set(xdims).intersection(set(ydims)):
    xdims = [dim + ("x" if dim in xydims else "") for dim in xdims]
    ydims = [dim + ("y" if dim in xydims else "") for dim in ydims]

    for dim in xydims:
        s = tensor.size(dim).sqrt().ceil().int()
        tensor = nn.ConstantPad1d((0, s**2 - tensor.size(dim)), float("nan"))(tensor)
        tensor = tensor.unflatten(dim, ((dim + "x", s), (dim + "y", s)))

    matrix = tensor.flatten(xdims, "x").flatten(ydims, "y")

    image = plt.matshow(matrix, fignum=fignum)
    plt.colorbar()
    return image
