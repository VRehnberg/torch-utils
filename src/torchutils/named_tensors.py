import functools
from collections import Counter, defaultdict
from itertools import starmap
from functools import partial
import itertools
from typing import Callable, Iterable
from itertools import chain
from lenses import lens
import opt_einsum

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


def naive_nmap(func: Callable, name):
    def vfunc(inputs):
        return nstack(*(
            func(input) for input in inputs.unbind(name)
        ), name=name)
    return vfunc


vmap = partial(nmap, functorch.vmap)
    

def index(input, s, name):
    if hasattr(s, "names"):
        s = s.rename(None)
    slices = [s if n == name else slice(None) for n in input.names]
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
        names = chain(*(renames.get(k, [k]) for k in names))
        return lens.Instance(torch.Tensor).call_refine_names(*names, ...)(output)

    return wrapped


def nstack(*tensors, name=None):
    return lift_nameless(torch.stack)(tensors, dim=0).refine_names(name, ...)


torch.Tensor.nsize = lambda self: {n: s for n, s in zip(self.names, self.shape)}


def GA(attr):
    def setter(x, y):
        x.__setattr__(attr, y)
        return x
    return lens.Lens(lambda x: x.__getattribute__(attr), setter)


def neinsum(input, other, **instructions):
    
    other = other.rename(**{
        name: name + "1" for name, i in instructions.items() if i == 2
    })

    # matmul <=> abc, acd -> abd
    left = set(input.nsize().items())
    right = set(other.nsize().items())
    batch = {(n, input.size(n)) for n in instructions}
    a = dict(left & right & batch)
    b = dict(left - right)
    c = dict(left & right - batch)
    d = dict(right - left)
    abc = input.clone().flatten(tuple(a), "a").flatten(tuple(b), "b").flatten(tuple(c), "c").align_to("a", "b", "c")
    acd = other.clone().flatten(tuple(a), "a").flatten(tuple(c), "c").flatten(tuple(d), "d").align_to("a", "c", "d")
    abd = abc @ acd

    return abd.unflatten("a", tuple(a.items())).unflatten("b", tuple(b.items())).unflatten("d", tuple(d.items()))


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

torch.Tensor.nunsqueeze = unsqueeze
