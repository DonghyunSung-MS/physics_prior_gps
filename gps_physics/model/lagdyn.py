# modify from https://torchdyn.readthedocs.io/en/latest/tutorials/09_lagrangian_nets.html

import torch
import torch.nn as nn
import torchdyn
from torch.autograd import grad


class ControlledLNN(nn.Module):
    def __init__(self, L, x_dim, input_matrix=nn.Identity()):
        super().__init__()
        self.L = L  # model
        self.x_dim = x_dim
        self.input_matrix = input_matrix
        self.input_matrix.requires_grad_(False)

    def forward(self, xu):
        """Compute Largrangian Dynamics with action(external-force)

        assume constant forcing term

        q_dd = (DqdDqdL)^-1 @ (DqdL - DqDqdL*q_d + B*tau)

        Args:
            xu (torch.Tensor): shape (bs, x_dim + u_dim)

        Returns:
            [torch.Tensor]: shape (bs, x_dim + u_dim) bs x [q_d, q_dd, 0]
        """
        with torch.set_grad_enabled(True):
            self.n = n = self.x_dim // 2  # x (bs, 2D) D in q dim
            xu = xu.requires_grad_(True)

            qqd = xu[:, : self.x_dim]
            u = xu[:, self.x_dim :]

            L = self._lagrangian(qqd).sum()
            J = grad(L, qqd, create_graph=True)[0]
            DL_q, DL_qd = J[:, :n], J[:, n:]
            DDL_qd = []  # hessian
            for i in range(n):
                J_qd_i = DL_qd[:, i][:, None]
                H_i = grad(J_qd_i.sum(), qqd, create_graph=True)[0][:, :, None]
                DDL_qd.append(H_i)
            DDL_qd = torch.cat(DDL_qd, 2)
            DDL_qqd, DDL_qdqd = DDL_qd[:, :n, :], DDL_qd[:, n:, :]

            T = torch.einsum("ijk, ij -> ik", DDL_qqd, qqd[:, n:])  # (bs, q, qd) * (bs, qd)
            qdd = torch.einsum("ijk, ij -> ik", DDL_qdqd.pinverse(), DL_q - T + self.input_matrix(u))

        return torch.cat([qqd[:, self.n :], qdd, torch.zeros_like(u)], 1)

    def _lagrangian(self, qqd):
        return self.L(qqd)


class NonConsLNN(nn.Module):
    def __init__(self, L, Q, x_dim):
        super().__init__()
        self.L = L  # model
        self.Q = Q
        self.x_dim = x_dim
        # self.input_matrix.requires_grad_(False)

    def forward(self, xu):
        """Compute Largrangian Dynamics with action(external-force)

        assume constant forcing term

        q_dd = (DqdDqdL)^-1 @ (DqdL - DqDqdL*q_d + B*tau)

        Args:
            xu (torch.Tensor): shape (bs, x_dim + u_dim)

        Returns:
            [torch.Tensor]: shape (bs, x_dim + u_dim) bs x [q_d, q_dd, 0]
        """
        with torch.set_grad_enabled(True):
            self.n = n = self.x_dim // 2  # x (bs, 2D) D in q dim
            xu = xu.requires_grad_(True)

            qqd = xu[:, : self.x_dim]
            u = xu[:, self.x_dim :]

            L = self._lagrangian(qqd).sum()
            J = grad(L, qqd, create_graph=True)[0]
            DL_q, DL_qd = J[:, :n], J[:, n:]
            DDL_qd = []  # hessian
            for i in range(n):
                J_qd_i = DL_qd[:, i][:, None]
                H_i = grad(J_qd_i.sum(), qqd, create_graph=True)[0][:, :, None]
                DDL_qd.append(H_i)
            DDL_qd = torch.cat(DDL_qd, 2)
            DDL_qqd, DDL_qdqd = DDL_qd[:, :n, :], DDL_qd[:, n:, :]

            T = torch.einsum("ijk, ij -> ik", DDL_qqd, qqd[:, n:])  # (bs, q, qd) * (bs, qd)
            qdd = torch.einsum("ijk, ij -> ik", DDL_qdqd.pinverse(), DL_q - T + self.Q(xu))

        return torch.cat([qqd[:, self.n :], qdd, torch.zeros_like(u)], 1)

    def _lagrangian(self, qqd):
        return self.L(qqd)
