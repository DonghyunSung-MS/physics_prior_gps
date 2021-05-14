import torch
from torch import nn


def mlp(input_size: int, hidden_size: int, output_size: int, layer_depths: int, act: str, out_act: str = "identity"):
    layers = []
    for i in range(layer_depths):
        ac = string2activation(act) if i < layer_depths - 1 else string2activation(out_act)
        if i == 0:
            layers += [nn.Linear(input_size, hidden_size), ac()]

        elif i == layer_depths - 1:
            layers += [nn.Linear(hidden_size, output_size), ac()]
        else:
            layers += [nn.Linear(hidden_size, hidden_size), ac()]

    return nn.Sequential(*layers)


def string2activation(act: str):
    activation = None
    if act == "relu":
        activation = nn.ReLU
    elif act == "identity":
        activation = nn.Identity
    elif act == "tanh":
        activation = nn.Tanh
    elif act == "softplus":
        activation = nn.Softplus
    elif act == "elu":
        activation = nn.ELU
    else:
        raise NotImplementedError(f"{act} is not implemeted")

    return activation


class CosSin(nn.Module):
    """ref: https://github.com/mfinzi/constrained-hamiltonian-neural-networks"""

    def __init__(self, input_dim, angular_dims):
        super(CosSin, self).__init__()
        self.input_dim = input_dim
        self.angular_dims = torch.LongTensor(angular_dims)
        self.non_angular_dims = torch.LongTensor(list(set(range(input_dim)) - set(angular_dims)))

    def forward(self, xu):
        ang_q = xu[:, self.angular_dims]
        other = xu[:, self.non_angular_dims]

        cos_ang_q, sin_ang_q = torch.cos(ang_q), torch.sin(ang_q)
        xu_tansform = torch.cat([cos_ang_q, sin_ang_q, other], dim=-1)

        return xu_tansform
