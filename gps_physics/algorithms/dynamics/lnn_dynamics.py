from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as ptdata
from numpy.lib.twodim_base import eye
from torch.autograd.functional import jacobian
from torchdyn.models import NeuralDE

from gps_physics.algorithms.dynamics.dynamics import Dynamics
from gps_physics.model.lagdyn import ControlledLNN, NonConsLNN
from gps_physics.utils.dyn_train import LNNLearner
from gps_physics.utils.ptu import *
from gps_physics.utils.samples import Trajectory


class LNNprior:
    def __init__(self, x_dim, u_dim, dt, hyperparams):
        self._hyperparams = hyperparams
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        L = torch.nn.Sequential(
            *[
                CosSin(x_dim, angular_dims=hyperparams["angular_dims"]),
                nn.Linear(x_dim + len(hyperparams["angular_dims"]), hyperparams["hidden_size"]),
                nn.Softplus(),
                nn.Linear(hyperparams["hidden_size"], hyperparams["hidden_size"]),
                nn.Softplus(),
                nn.Linear(hyperparams["hidden_size"], 1),
            ]
        )
        Q = torch.nn.Sequential(
            *[
                CosSin(x_dim, angular_dims=hyperparams["angular_dims"]),
                nn.Linear(x_dim + len(hyperparams["angular_dims"]), hyperparams["hidden_size"]),
                nn.Softplus(),
                # nn.Linear(hyperparams["hidden_size"], hyperparams["hidden_size"]),
                # nn.Softplus(),
                nn.Linear(hyperparams["hidden_size"], x_dim // 2),
            ]
        )
        # L = torch.nn.Sequential(
        #     *[
        #         # CosSin(x_dim, angular_dims=hyperparams["angular_dims"]),
        #         nn.Linear(x_dim, hyperparams["hidden_size"]),
        #         nn.Softplus(),
        #         nn.Linear(hyperparams["hidden_size"], hyperparams["hidden_size"]),
        #         nn.Softplus(),
        #         nn.Linear(hyperparams["hidden_size"], 1),
        #     ]
        # )
        # lnn = ControlledLNN(L, x_dim)
        lnn = NonConsLNN(L, Q, x_dim)

        self.model = NeuralDE(func=lnn, solver="dopri8")  # torch.nn.Module
        self.learner = LNNLearner(self.model, self.dt, self.x_dim, hyperparams["lr"])  # pl moudle

    def fit(self, xu: np.array, dxdu: np.array):
        print(xu.shape, dxdu.shape)
        traindata = ptdata.TensorDataset(torch.FloatTensor(xu), torch.FloatTensor(dxdu))
        trainloader = ptdata.DataLoader(
            traindata, self._hyperparams["batch_size"], shuffle=True, num_workers=0, drop_last=True
        )
        self.learner.fit(self._hyperparams["epoch"], trainloader)


class DynamicsLRLNN(Dynamics):
    def __init__(self, x_dim, u_dim, dt, hyperparams):
        Dynamics.__init__(self, hyperparams)

        self.prior = LNNprior(x_dim, u_dim, dt, hyperparams["prior"])
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt = dt

    def updata_prior(self, x_xu: List[np.array]):
        """Construct Data Loader for physic prior

        x' x u to dx = (x' - x)/dt, du = 0

        Args:
            xux (List[Trajectory]):
            list index -> global iteration, each iteration contains trajectory data[M, N, T, x'xu]


        """
        # print(x_xu.shape)
        x_xu = np.stack(x_xu).reshape(-1, self.x_dim * 2 + self.u_dim)

        MNT = x_xu.shape[0]
        nq = self.x_dim // 2

        qdd = (x_xu[:, nq : 2 * nq] - x_xu[:, self.x_dim + nq : self.x_dim + 2 * nq]) / self.dt
        qd = x_xu[:, self.x_dim + nq : self.x_dim + 2 * nq]

        # dx = (x_xu[:, :self.x_dim] - x_xu[:, self.x_dim:2*self.x_dim])/self.dt
        dx = np.hstack([qd, qdd])

        # du = (x_xu[:, 2*self.x_dim:] - x_xu[:, 2*self.x_dim:]) / self.dt
        du = np.zeros((MNT, self.u_dim))

        dxdu = np.hstack([dx, du])
        xu = x_xu[:, self.x_dim :]

        self.prior.fit(xu, dxdu)
        # self.prior.fit(xu, x_xu[:, :self.x_dim])

    def fit(self, mean_traj: np.array):
        """Linearized dynamics along mean trajectory

        Args:
            mean_traj (np.array): T * x u
        """
        T = mean_traj.shape[0]

        self.AB_t = np.zeros((T, self.x_dim, self.x_dim + self.u_dim))
        self.c_t = np.zeros((T, self.x_dim))
        self.W_t = np.zeros((T, self.x_dim, self.x_dim))

        q = self.x_dim // 2

        for t in range(T):
            xu = torch.FloatTensor(mean_traj[t]).reshape(1, -1)

            D_qqd_qdd = jacobian(self.model.defunc.m.forward, xu).squeeze()[q : 2 * q].detach().numpy()
            f_star = self.model.defunc(0, xu).squeeze()[q : 2 * q].detach().numpy()

            res = f_star - D_qqd_qdd @ mean_traj[t]

            ident = np.block([[np.eye(q), np.eye(q) * self.dt], [np.zeros(self.x_dim, self.x_dim), np.eye(q)]])

            A_con = D_qqd_qdd[:, : 2 * q]
            B_con = D_qqd_qdd[:, 2 * q :]

            A_disc = np.block(
                [[A_con * self.dt ** 2, np.zeros(self.x_dim, self.x_dim)], [np.zeros(self.x_dim, self.x_dim), A_con * self.dt]]
            )

            A_disc = A_disc + ident
            B_disc = np.block([[B_con * self.dt ** 2], [B_con * self.dt]])

            c_disc = np.block([[res * self.dt ** 2], [res * self.dt]])

            self.AB_t[t] = np.hstack([A_disc, B_disc])
            self.c_t[t] = c_disc
            self.W_t[t] = 0.1 * np.eye(self.x_dim)  # TODO better noise reprenstation

        return self.AB_t, self.c_t, self.W_t
