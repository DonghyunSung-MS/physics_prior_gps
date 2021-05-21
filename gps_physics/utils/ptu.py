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
        self.der = None

    def forward(self, xu):
        ang_q = xu[:, self.angular_dims]
        other = xu[:, self.non_angular_dims]

        cos_ang_q, sin_ang_q = torch.cos(ang_q), torch.sin(ang_q)
        xu_tansform = torch.cat([cos_ang_q, sin_ang_q, other], dim=-1)

        if other.shape[1] == 0:
            self.der = torch.cat([torch.diag_embed(-sin_ang_q), torch.diag_embed(cos_ang_q)], dim=1)
        else:
            self.der = torch.cat(
                [torch.diag_embed(-sin_ang_q), torch.diag_embed(cos_ang_q), torch.diag_embed(torch.ones_like(other))], dim=1
            )

        return xu_tansform


class ConstScaleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        return input * self.scale


class InputMatrixLayer(nn.Module):
    def __init__(self, q_dim, u_dim, input_mat):
        super().__init__()
        self.input_mat = nn.Linear(u_dim, q_dim, False)
        print(self.input_mat.weight.shape)

        self.input_mat.weight = nn.Parameter(torch.FloatTensor(input_mat))
        self.input_mat.requires_grad_(False)
        print(self.input_mat.weight.shape)

    def forward(self, input):
        return self.input_mat(input)


if __name__ == "__main__":
    import numpy as np
    from torch.autograd.functional import jacobian

    # input = torch.ones(1, 1) * 3.14
    # cossin = CosSin(1, [0])
    # print(cossin(input))
    # print(cossin.der)
    # out = torch.zeros(10, 12, 12) @ torch.zeros(12, 2) @ torch.zeros(10, 2, 1)
    # print(out.shape)
    # input_mat = InputMatrixLayer(2, 1, input_mat=np.array([[0.0], [1.0]]))
    # print(input_mat(torch.randn(1)))

    print(torch.diag_embed(torch.ones(1, 2)))
    print(torch.diag(torch.ones(1, 2)))
