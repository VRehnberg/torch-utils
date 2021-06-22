import functools
from functools import partial
from typing import Callable, Iterable

import torch
import functorch


def nmap(mapper, func : Callable, over_dims):
    over_dims = [over_dims] if not isinstance(over_dims, Iterable) else over_dims

    in_names = None
    out_names = None
    @functools.wraps(func)
    def nfunc(input):
        nonlocal in_names
        nonlocal out_names
        input.names = in_names
        output = func(input)
        out_names = output.names
        return output

    @functools.wraps(nfunc)
    def vfunc(inputs):
        inputs = inputs.align_to(*over_dims, ...)
        over_names = inputs.names[len(over_dims):]
        in_names = inputs.names[:len(over_dims)]
        dims = tuple(range(len(over_dims)))
        outputs = mapper(nfunc, dims, dims)(inputs)
        outputs.refine_names(*over_names, *out_names)
        return outputs

    return vfunc


def naive_nmap(func: Callable, dim):
    def vfunc(inputs):
        return torch.stack([
            func(input) for input in inputs.unbind(dim)
        ]).refine_names(dim, ...)
    return vfunc


vmap = partial(nmap, functorch.vmap)
    

def index(input, s, dim):
    if hasattr(s, "names"):
        s = s.rename(None)
    slices = [s if name == dim else slice(None) for name in input.names]
    return input.rename(None)[slices].rename(*input.names)


torch.Tensor.index = index


def unify(input, other):
    if input is None:
        return other
    assert len(input) == len(other)
    def name_unify(i, o):
        if i is None: return o
        if o is None: return i
        assert i==o
        return i
    return [name_unify(i, o) for i, o in zip(input, other)]


def lift_nameless(func, **renames):
    def wrapped(*args, **kwargs):
        def traverse(f, input):
            if isinstance(input, torch.Tensor):
                f(input)
            elif isinstance(input, dict):
                for v in input.values():
                    traverse(f, v)
            elif isinstance(input, Iterable):
                for i in input:
                    traverse(f, i)
        
        names = None
        def strip_name(input):
            nonlocal names
            names = unify(names, input.names)
            input._tmp_names = input.names
            input.rename_(None)
        
        def return_name(input):
            input.rename_(*input._tmp_names)
        
        traverse(strip_name, (args, kwargs))
        output = func(*args, **kwargs)
        traverse(return_name , (args, kwargs))
        names = [v for k in names for v in (renames[k] if k in renames else [k])]
        return output.refine_names(*names, ...)
    return wrapped
    