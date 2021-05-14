import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad
from torch.autograd.functional import hessian, jacobian

### scalar function ####


def scalar_func(x):
    return x ** 2 + x


print(jacobian(scalar_func, torch.ones(1), create_graph=True))  # f'(x) = 2x + 1
print(hessian(scalar_func, torch.ones(1), create_graph=True))  # f''(x) = 2

### vector quadratic function ####
def vector_func(x):
    H = torch.FloatTensor([[1.0, -1.0], [-1.0, 2.0]])
    g = torch.FloatTensor([3.0, 1.0])
    return 0.5 * x.t().matmul(H).matmul(x) + g.t().matmul(x)


### neural network function ####
simple_model = nn.Sequential(*[nn.Linear(4, 2), nn.Softplus(), nn.Linear(2, 1)])

x = torch.ones(4)
x.requires_grad = True
y = simple_model(x)

# print(grad(y, x, retain_graph=True, create_graph=True))
# print(jacobian(simple_model, x, create_graph=True)) # f'(x) = 2x + 1

# print(hessian(simple_model, x, create_graph=True).shape)

""" Comparing pytorch and jax for calculating diagonal vector of Hessian matrix
"""
# pytorch

import torch
import torch.nn as nn


def pth_jacobian(y, x, create_graph=False):
    """
    reference: https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        (grad_x,) = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        grad_x = grad_x.reshape(x.shape)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0

    return torch.stack(jac, axis=0).reshape(y.shape + x.shape)


def pth_hessian(y, x):
    return pth_jacobian(pth_jacobian(y, x, create_graph=True), x)


def pth_hessian_with_loop(y, x):
    Hs = []
    batch_size = y.size(0)
    for i in range(batch_size):
        H = pth_jacobian(pth_jacobian(y[i], x[i], create_graph=True), x)

    return torch.stack(H, axis=0)


def pth_hessian_diag(y, x):
    H = pth_hessian(y, x)
    batch_size = y.size(0)

    diag_vec = []
    for i in range(batch_size):
        diag_vec.append(H[i, :, i, :, i, :])

    diag = torch.stack(diag_vec, dim=0)
    x_dim = x.size(1)

    diag_vec = []
    for i in range(x_dim):
        diag_vec.append(diag[:, :, i, i])

    diag = torch.stack(diag_vec, dim=-1)
    return diag


class FC(nn.Module):
    def __init__(self, nc_in, nc_out, num_channels):
        super().__init__()

        if not isinstance(num_channels, list):
            num_channels = [num_channels]

        modules = []
        self.nc_in = nc_in
        self.nc_out = nc_out

        for nc in num_channels:
            modules.append(nn.Linear(nc_in, nc))
            modules.append(nn.Sigmoid())
            nc_in = nc

        modules.append(nn.Linear(nc_in, nc_out))
        modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


import time

import jax
import jax.nn as jnn

# jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit, vmap


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale=1) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def create_network_params(in_nc, out_nc, channels=[]):
    return init_network_params([in_nc] + channels + [out_nc], jax.random.PRNGKey(0))


def predict(params, input):
    activations = input

    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = sigmoid(outputs)

    final_w, final_b = params[-1]
    output = jnp.dot(final_w, activations) + final_b
    output = sigmoid(output)
    return output


def sigmoid(x):
    return jnn.sigmoid(x)


def jax_hessian(f):
    # we are using double jacfwd.
    # jacfwd is more efficient for 'tall' jac matrices.
    # jacrev is more efficient for 'wide' jac matrices.
    # in our case, input is mostly tall, not wide.
    # so jacfwd(jacfwd(x)) is faster than jacfwd(jacrev)
    return jit(jacfwd(jacfwd(f)))


def jax_hessian_diag(f, input):
    H = jax_hessian(f)(input)
    diag = H.diagonal(0, 1, 2)
    return diag


vjax_hessian_diag = vmap(jax_hessian_diag, in_axes=(None, 0))


def main():
    batch_size = 32
    print("Batch size: ", batch_size)
    # pytorch main
    print("Run pytorch")
    x = torch.rand(batch_size, 3).requires_grad_(True).to("cpu")
    torch_net = FC(3, 1, [256]).to("cpu")
    y = torch_net(x) ** 1  # trick: https://discuss.pytorch.org/t/hessian-of-output-with-respect-to-inputs/60668/10

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    torch_hd = pth_hessian_diag(y, x)
    end.record()

    torch.cuda.synchronize()
    print("pytorch elapsed: {}ms".format(start.elapsed_time(end)))
    torch_hd = torch_hd.cpu().numpy()
    x = x.detach().cpu().numpy()
    torch.cuda.empty_cache()

    # jax main (fast test version)

    print("Run jax")

    # x = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, 3))
    x = jnp.array(x)  # use same x

    params = create_network_params(3, 1, [256])
    net = lambda x: predict(params, x)
    vnet = vmap(net)
    y = vnet(x)

    start_ts = time.time()
    jax_hd = vjax_hessian_diag(net, x)
    end_ts = time.time()

    print("jax elapesed: {}ms".format((end_ts - start_ts) * 1000.0))

    print("Run jax validation")
    # jax 2 (slow accurate version)

    start_ts = time.time()
    accurate_H = jax_hessian(vnet)(x)
    H_per_row = []
    for i in range(batch_size):
        d = accurate_H[i, :, i, :, i, :]
        H_per_row.append(d)
    H_per_row = jnp.stack(H_per_row, axis=0)
    accurate_h_diag = H_per_row.diagonal(0, 2, 3)
    end_ts = time.time()

    is_fast_jax_correct = jnp.allclose(accurate_h_diag, jax_hd)
    print("is fast jax version correct:", is_fast_jax_correct)
    print("jax validation elapesed: {}ms".format((end_ts - start_ts) * 1000.0))


if __name__ == "__main__":
    main()
