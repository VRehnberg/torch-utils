from functools import partial, wraps
from itertools import chain
from lenses import lens, bind
import lenses.hooks
from lenses.optics import Setter
import torch


def nstack(*tensors, name=None):
    return lift_nameless(torch.stack)(tensors, dim=0).refine_names(name, ...)

over = lambda name: lens.Iso(lambda t:t.unbind(name), lambda ts:nstack(*ts, name=name)).Each()
torch.Tensor.over = lambda self, name: bind(self) & over(name)
    

def index(input, s, name):
    return lift_nameless(lambda t, slices:t[slices])(input,
        [s if n == name else slice(None) for n in input.names])

torch.Tensor.index = index


@lenses.hooks.setattr.register
def _(self: torch.Tensor, name, value):
    setattr(self, name, value)
    return self # Cheating: Mutates in-place!


def lift_nameless(func, out=None, add=None, **renames):
    @wraps(func)
    def wrapped(*args, **kwargs):
        namess = bind([args, kwargs]).Recur(torch.Tensor).names
        saved = namess.collect()
        namess.set(None)
        output = func(*args, **kwargs)
        namess.set_many(saved)
        names = saved[0] if out is None else out
        names = chain(*(renames.get(k, [k]) for k in names), bind(add).Recur(str).collect())
        return bind(output).Instance(torch.Tensor).call_refine_names(*names)
    return wrapped

class Nameless(Setter):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args
    def func(self, f, state):
        return lift_nameless(f, *args, **kwargs)(state, *args, **kwargs)


def neinsum(input, other, **instructions):
    renames = {n: n + "1" for n, i in instructions.items() if i == 2}
    other = other.rename(**renames) if renames else other

    # matmul <=> abc, acd -> abd
    left = set(zip(input.names, input.shape))
    right = set(zip(other.names, other.shape))
    batch = {(n, input.size(n)) for n, i in instructions.items() if i == 1}
    a = left & right & batch
    b = left - right
    c = left & right - batch
    d = right - left
    abc = input.nflatten(a=a,b=b,c=c)
    acd = other.nflatten(a=a,c=c,d=d)
    return (abc @ acd).nunflatten(a=a,b=b,d=d)


def nflatten(self, **kwargs):
    for name,olds in kwargs.items():
        olds = tuple(bind(olds).Recur(str).collect())
        self = self.align_to(..., *olds).flatten(olds, name) if olds else self.rename(None).unsqueeze(-1).rename(*self.names, name)
    return self

def nunflatten(self, **kwargs):
    for name,news in kwargs.items():
        news = tuple(bind(news).Each().collect())
        self = self.unflatten(name, news) if news else self.squeeze(name)
    return self

torch.Tensor.nflatten = nflatten
torch.Tensor.nunflatten = nunflatten


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
