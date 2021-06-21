from typing import Callable, Iterable
import functools
import functorch


def vmap(func : Callable, over_dims):
    '''
        over_dims : 
    '''
    over_dims = [over_dims] if not isinstance(over_dims, Iterable) else over_dims

    in_names = None
    out_names = None
    @functools.wraps(func)
    def nfunc(input):
        nonlocal in_names
        nonlocal out_names
        output = func(input)
        out_names = output.names
        return output

    @functools.wraps(nfunc)
    def vfunc(inputs):
        inputs = inputs.align_to(*over_dims, ...)
        over_names = inputs.names[len(over_dims):]
        in_names = inputs.names[:len(over_dims)]
        dims = tuple(range(len(over_dims)))
        outputs = functorch.vmap(nfunc, dims, dims)(inputs)
        outputs.refine_names(*over_names, *out_names)
        return outputs

    return vfunc
    