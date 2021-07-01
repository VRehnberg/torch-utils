import functools
from collections import Counter, defaultdict
from itertools import starmap
from functools import partial
import itertools
from typing import Callable, Iterable
from itertools import chain
from lenses import lens

import torch
import functorch


COMMON_ITERABLES = (tuple, list, dict)


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


def lift_nameless(func, out_names=None, **renames):
    def wrapped(*args, **kwargs):
        def traverse(f, input):
            lens.Instance(torch.Tensor).modify(f)(input)
            lens.Fork(
                lens.Instance((list, tuple)).Each(),
                lens.Instance(dict).Each()[1],
            ).modify(partial(traverse, f))(input)
        
        names = out_names

        def save_name(input):
            if out_names is None:
                nonlocal names
                names = unify(names, input.names)
            input._tmp_names = input.names

        def strip_name(input):
            input.rename_(None)
        
        def return_name(input):
            input.rename_(*input._tmp_names)
        
        traverse(save_name, [args, kwargs])
        traverse(strip_name, [args, kwargs])
        output = func(*args, **kwargs)
        traverse(return_name, [args, kwargs])
        names = tuple(chain(*(renames.get(k, [k]) for k in names)))
        return output.refine_names(*names, ...) if isinstance(output, torch.Tensor) else output

    return wrapped
    

def neinsum(*tensors, **instructions):
    
    alphabet = (chr(i) for i in itertools.count(97))

    ins = [t.names for t in tensors]
    def yield_outs():
        nonlocal ins
        counts = Counter(chain(*ins))
        suffixes = [""] + [str(i) for i in range(1, sum(counts.values()))]
        for name, count in counts.items():
            n_out = instructions.get(name, max(0, 1 - count))
            cs = list(itertools.islice(alphabet, count)) if n_out > 1 else [next(alphabet)] * count
            ins = lens.Each().Each().Filter(lambda n: n == name).set_many(cs)(ins)
            yield from zip(cs[:n_out], [name+suffix for suffix in suffixes])
    outs, outnames = zip(*yield_outs())
    operation = ",".join(["".join(ns) for ns in ins]) + "->" + "".join(outs)

    return lift_nameless(torch.einsum, out_names=outnames)(operation, *tensors)


def ndiagonal(input, offset=0, **join_names):
    if len(join_names)==0:
        return input
    
    name1 = next(iter(join_names))
    name2 = join_names.pop(name1)

    out_names = [name for name in input.names if name not in [name1, name2]] + [name1]

    input_names = input.names
    def get_dim(name):
        return input_names.index(name)

    output = input.diagonal(offset=offset, dim1=get_dim(name1), dim2=get_dim(name2)).rename(*out_names)
    return ndiagonal(output, offset=offset, **join_names)


def unsqueeze(input, dim, name=None):
    out_names = input.names[:dim] + (name,) + input.names[dim:]
    return input.rename(None).unsqueeze(dim).rename(*out_names)
