"""
Ref 
https://torchdyn.readthedocs.io/en/latest/tutorials/09_lagrangian_nets.html
https://github.com/powertj/EECS545_Project_DeLaN.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdyn
from torch.autograd import grad

from gps_physics.utils.ptu import CosSin


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
            # qdd = torch.einsum("ijk, ij -> ik", DDL_qdqd.pinverse(), DL_q - T + self.Q(xu))
            qdd = torch.einsum("ijk, ij -> ik", DDL_qdqd.inverse(), DL_q - T + u)

        return torch.cat([qqd[:, self.n :], qdd, torch.zeros_like(u)], 1)

    def _lagrangian(self, qqd):
        return self.L(qqd)


class DeepLagrangianNetwork(nn.Module):
    def __init__(self, q_dim, hidden_dim=64, device="cpu", angular_dims=None):
        if angular_dims == None:
            angular_dims = range(q_dim)
        super().__init__()
        self.q_dim = q_dim
        self.num_Lo = int(0.5 * (q_dim ** 2 - q_dim))
        # consin embeding
        self.cos_sin_embdeing = CosSin(q_dim, angular_dims)
        self.fc1 = nn.Linear(q_dim + len(angular_dims), hidden_dim)
        # self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.fc_G = nn.Linear(hidden_dim, q_dim)
        self.fc_Ld = nn.Linear(hidden_dim, q_dim)
        self.fc_Lo = nn.Linear(hidden_dim, self.num_Lo)

        self.act_fn = F.leaky_relu
        self.neg_slope = -0.01
        self.device = device
        self.interim_values = {}

    def compute_gradients_for_forward_pass(self, qdot, h1, h2, h3):
        """
        Computes partial derivatives of the inertia matrix (H) needed for the forward pass
        :return: dHdq and dHdt
        """
        n, d = qdot.shape

        dRelu_fc1 = torch.where(
            h1 > 0, torch.ones(h1.shape, device=self.device), self.neg_slope * torch.ones(h1.shape, device=self.device)
        )
        # print(torch.diag_embed(dRelu_fc1).shape, self.fc1.weight.shape)
        dh1_dq = torch.diag_embed(dRelu_fc1) @ self.fc1.weight @ self.cos_sin_embdeing.der.unsqueeze(2)
        # print(dh1_dq.shape)

        dRelu_fc2 = torch.where(
            h2 > 0, torch.ones(h2.shape, device=self.device), self.neg_slope * torch.ones(h2.shape, device=self.device)
        )
        dh2_dh1 = torch.diag_embed(dRelu_fc2) @ self.fc2.weight

        dRelu_dfc_Ld = torch.sigmoid(h3)  # torch.where(ld > 0, torch.ones(ld.shape), 0.0 * torch.ones(ld.shape))

        dld_dh2 = torch.diag_embed(dRelu_dfc_Ld) @ self.fc_Ld.weight
        dlo_dh2 = self.fc_Lo.weight

        dld_dq = dld_dh2 @ dh2_dh1 @ dh1_dq
        dlo_dq = dlo_dh2 @ dh2_dh1 @ dh1_dq

        dld_dt = (dld_dq @ qdot.view(n, d, 1)).squeeze(-1)
        dlo_dt = (dlo_dq @ qdot.view(n, d, 1)).squeeze(-1)
        dld_dq = dld_dq.permute(0, 2, 1)

        dL_dt = self.assemble_lower_triangular_matrix(dlo_dt, dld_dt)
        dL_dq = self.assemble_lower_triangular_matrix(dlo_dq, dld_dq)

        return dL_dq, dL_dt

    def assemble_lower_triangular_matrix(self, Lo, Ld):
        """
        Assembled a lower triangular matrix from it's diagonal and off-diagonal elements
        :param Lo: Off diagonal elements of lower triangular matrix
        :param Ld: Diagonal elements of lower triangular matrix
        :return: Lower triangular matrix L
        """
        assert 2 * Lo.shape[1] == (Ld.shape[1] ** 2 - Ld.shape[1])

        diagonal_matrix = torch.diag_embed(Ld)
        L = torch.tril(torch.ones(*diagonal_matrix.shape, device=self.device)) - torch.eye(self.q_dim)

        # Set off diagonals
        L[L == 1] = Lo.view(-1)
        # Add diagonals
        L = L + diagonal_matrix
        return L

    def inverse_dyn(self, x):
        """
        Deep Lagrangian Network inverse action model forward pass
        :param x: State consisting of q, q_dot, q_ddot
        :return: tau - action, H @ q_ddot, C, G
        where H is inertia matrix, C coriolis term, G is potentials term
        """
        q, q_dot, q_ddot = torch.chunk(x, chunks=3, dim=1)
        n, d = q.shape

        hidden1 = self.act_fn(self.fc1(self.cos_sin_embdeing(q)))
        # hidden1 = self.act_fn(self.fc1(q))
        hidden2 = self.act_fn(self.fc2(hidden1))
        hidden3 = self.fc_Ld(hidden2)

        g = self.fc_G(hidden2)
        Ld = F.softplus(hidden3)
        Lo = self.fc_Lo(hidden2)
        L = self.assemble_lower_triangular_matrix(Lo, Ld)
        dL_dq, dL_dt = self.compute_gradients_for_forward_pass(q_dot, hidden1, hidden2, hidden3)

        # Inertia matrix and time derivative
        H = L @ L.transpose(1, 2) + 1e-9 * torch.eye(d, device=self.device)
        dH_dt = L @ dL_dt.permute(0, 2, 1) + dL_dt @ L.permute(0, 2, 1)

        # Compute quadratic term d/dq [q_dot.T @ H @ q_dot]
        q_dot_repeated = q_dot.repeat(d, 1)
        dL_dqi = dL_dq.view(n * d, d, d)
        L_repeated = L.repeat(d, 1, 1)
        quadratic_term = (
            q_dot_repeated.view(-1, 1, d)
            @ (dL_dqi @ L_repeated.transpose(1, 2) + L_repeated @ dL_dqi.transpose(1, 2))
            @ q_dot_repeated.view(-1, d, 1)
        )

        # Compute coriolis term
        c = dH_dt @ q_dot.view(n, d, 1) - 0.5 * quadratic_term.view(n, d, 1)
        tau = H @ q_ddot.view(n, d, 1) + c + g.view(n, d, 1)

        return tau.reshape(n, d), H.reshape(n, d, d), c.reshape(n, d), g.reshape(n, d)

    def forward(self, xu):
        q, q_dot, u = torch.chunk(xu, chunks=3, dim=1)
        qddot = torch.zeros_like(u)
        q_qd_qdd = torch.cat([q, q_dot, qddot], dim=1)

        tau, H, c, g = self.inverse_dyn(q_qd_qdd)
        qddot_pred = torch.einsum("ijk, ij -> ik", H.inverse(), (u - c - g))

        return torch.cat([q_dot, qddot_pred, torch.zeros_like(u)], dim=1)

    def discrete_predict(self, xu):
        def func(order, xu):
            return self.forward(xu)

        sol = torchdyn.odeint(func, xu, torch.linspace(0, 0.01, 10))

        return sol[-1]


if __name__ == "__main__":
    from torchdyn.models import NeuralDE

    n_dof = 2
    batch = 2
    network = DeepLagrangianNetwork(n_dof, 64)

    model = NeuralDE(func=network, solver="dopri5")

    test_input = torch.ones(batch, n_dof * 3)
    print(model.defunc(0, test_input))
    print(network(test_input))
