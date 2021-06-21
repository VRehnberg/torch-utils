import gc
import torch
from torch import nn
from torch.autograd.functional import jacobian
    
def get_activations(network, x, from_module_types=None):
    if from_module_types is None:
        from_module_types = []
    elif isinstance(from_module_types, nn.Module):
        from_module_types = [from_module_types]
    def check_module_type(module):
        return any(isinstance(module, module_type) for module_type in from_module_types)

    activations = []
    hooks = []
    save_activations = lambda mod, inp, out: activations.append(out)
    for name, m in network.named_modules():
        if check_module_type(m):
            hooks.append(m.register_forward_hook(save_activations))
    
    network(x)
    for h in hooks:
        h.remove()
    
    return torch.hstack(activations)


def get_linear_activations(network, x):
    return get_activations(from_module_types=nn.Linear)


def batched_jacobian(func, x, **kwargs):
    device = x.device

    # Compute batched Jacobian
    new_func = lambda x: func(x).sum(0)
    jac = jacobian(new_func, x, **kwargs)

    # Move batch dimension first
    dims = torch.arange(jac.ndim, device=device)
    batch_dim = dims[-x.ndim]
    jac = jac.movedim(list(dims[:batch_dim + 1]), [batch_dim, *dims[:batch_dim]])

    return jac


def clean_mem(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
