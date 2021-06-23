import functools
from collections import Counter, defaultdict
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


def lift_nameless(func, out_names=None, **renames):
    def wrapped(*args, **kwargs):
        def traverse(f, input):
            if isinstance(input, torch.Tensor):
                f(input)
            elif isinstance(input, dict):
                for v in input.values():
                    traverse(f, v)
            elif isinstance(input, Iterable) and (input not in input):
                for i in input:
                    traverse(f, i)

        names = out_names
        def strip_name(input):
            if out_names is None:
                nonlocal names
                names = unify(names, input.names)
            input._tmp_names = input.names
            input.rename_(None)
        
        def return_name(input):
            input.rename_(*input._tmp_names)
        
        traverse(strip_name, (args, kwargs))
        output = func(*args, **kwargs)
        traverse(return_name , (args, kwargs))
        names = [v for k in names for v in (rename[k] if k in renames else [k])]
        return output.refine_names(*names, ...)
    return wrapped
    

def neinsum(*tensors, **overrides):
    
    i = -1
    def next_chr():
        nonlocal i
        i += 1
        return chr(97 + i)

    all_names = Counter([name for tensor in tensors for name in tensor.names])
    instructions = {k: max(1 - v, 0) for k, v in all_names.items()}
    instructions.update(overrides)
    name2chr = {} #TODO

    out_names = []
    ins = len(tensors) * [""]
    outs = ""
    name2chr = {}
    out_uses = Counter()
    for i_tensor, tensor in enumerate(tensors):
        for name in tensor.names:
            instruction = instructions.get(name, 0)
            if instruction <= 1:
                name2chr[name] = next_chr()
    for i_tensor, tensor in enumerate(tensors):
        for name in tensor.names:
            this_chr = name2chr.get(name, next_chr())
            ins[i_tensor] += this_chr
            instruction = instructions.get(name, 0)
            if instruction > 0:
                outs += this_chr
                out_uses.update([name])
                uses = out_uses[name]
                out_names.append(f"{name}{uses if uses > 1 else ''}")

    name2chr = {name: chr(97 + i) for i, name in enumerate(all_names)}
    names2indices = lambda names: "".join([name2chr[name] for name in names])

    out_name2chr = {
        f'{name}{i if i > 0 else ""}': chr
        for name, chr in name2chr.items()
        for i in range(instructions.get(name, 1))
    }

    operation = (
        ",".join(names2indices(tensor.names) for tensor in tensors) +
        "->" + "".join(out_name2chr.values())
    )

    return lift_nameless(torch.einsum, out_names=tuple(out_name2chr))(operation, *tensors)


def _neinsum(input, other, out_names=None):
    '''If output_names is None then contract common names.'''
    
    assert None not in input.names
    assert None not in other.names

    if out_names is None:
        out_names = (
            [name for name in input.names if name not in other.names]
             + [name for name in other.names if name not in input.names]
        )
    
    all_names = set(input.names).union(set(other.names))
    name2chr = {name: chr(97 + i) for i, name in enumerate(all_names)}

    names2indices = lambda names: "".join([name2chr(name) for name in names])

    operation = f"{names2indices(input.names)},{names2indices(other.names)}->{out_names}"
    
    output = torch.einsum(operation, )


