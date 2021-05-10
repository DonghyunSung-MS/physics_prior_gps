import numpy as np

import torch
from torch import nn, optim
from torch.autograd.functional import jacobian, hessian

# simple_model = nn.Sequential(
#     *[nn.Linear(4, 2), nn.Softplus(), nn.Linear(2,1)]
# )

### scalar function ####

def scalar_func(x):
    return x**2 + x

### vector quadratic function ####
def vector_func(x):
    H = torch.FloatTensor([[1.0, -1.0], [-1.0, 2.0]])
    g = torch.FloatTensor([3.0, 1.0])
    return 0.5 * x.t().matmul(H).matmul(x) + g.t().matmul(x)

print(jacobian(scalar_func, torch.ones(1), create_graph=True)) # f'(x) = 2x + 1
print(hessian(scalar_func, torch.ones(1), create_graph=True)) # f''(x) = 2

print(jacobian(vector_func, torch.ones(2), create_graph=True)) # f'(x) = Hx + g
print(hessian(vector_func, torch.ones(2), create_graph=True)) # f''(x) = H