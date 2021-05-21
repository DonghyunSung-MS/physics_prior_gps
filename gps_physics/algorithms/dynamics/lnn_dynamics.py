import pickle
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as ptdata
from numpy.lib.twodim_base import eye
from torch.autograd.functional import jacobian
from torchdyn.models import NeuralDE

from gps_physics.algorithms.dynamics.dynamics import Dynamics
from gps_physics.model.lagdyn import ControlledLNN, DeepLagrangianNetwork, NonConsLNN
from gps_physics.utils.dyn_train import LNNLearner
from gps_physics.utils.ptu import *
from gps_physics.utils.samples import TrajectoryBuffer


class LNNprior:
    def __init__(self, x_dim, u_dim, dt, hyperparams):
        self._hyperparams = hyperparams
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim

        lnn = DeepLagrangianNetwork(
            x_dim // 2,
            hyperparams["hidden_size"],
            angular_dims=hyperparams["angular_dims"],
            input_matrix=InputMatrixLayer(x_dim // 2, u_dim, np.array(hyperparams["input_mat"])),
        )

        self.model = NeuralDE(func=lnn, solver="dopri8")  # torch.nn.Module
        self.learner = LNNLearner(self.model, self.dt, self.x_dim, hyperparams["lr"])  # pl moudle

    def fit(self, xu: np.array, dxdu: np.array):
        print(xu.shape, dxdu.shape)
        traindata = ptdata.TensorDataset(torch.FloatTensor(xu), torch.FloatTensor(dxdu))
        trainloader = ptdata.DataLoader(
            traindata, self._hyperparams["batch_size"], shuffle=False, num_workers=0, drop_last=True
        )
        self.learner.fit(self._hyperparams["epoch"], trainloader)


class DynamicsLRLNN(Dynamics):
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)

        self.x_dim = hyperparams["x_dim"]
        self.u_dim = hyperparams["u_dim"]
        self.dt = hyperparams["dt"]

        self.prior = LNNprior(self.x_dim, self.u_dim, self.dt, hyperparams["dyna_prior"])

    def updata_prior(self, x_xu: List[TrajectoryBuffer]):
        """Construct Data Loader for physic prior

        x' x u to dx = (x' - x)/dt, du = 0

        Args:
            xux (List[TrajectoryBuffer]):
            list index -> global iteration, each iteration contains trajectory data[M, N, T, x'xu]


        """
        # print(x_xu.shape)
        data = []
        for ele in x_xu:
            data.append(ele.traj)

        x_xu = np.stack(data).reshape(-1, self.x_dim * 2 + self.u_dim)

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

        perturbed equation delta dynamics

        Args:
            mean_traj (np.array): T * x' x u
        """
        T = mean_traj.shape[0]

        self.AB = np.zeros((T, self.x_dim, self.x_dim + self.u_dim))
        self.c = np.zeros((T, self.x_dim))
        self.W = np.zeros((T, self.x_dim, self.x_dim))

        q = self.x_dim // 2

        for t in range(T):
            xu = torch.FloatTensor(mean_traj[t][self.x_dim :]).reshape(1, -1)

            D_qqd_qdd = jacobian(self.prior.model.defunc.m.forward, xu).squeeze()[q : 2 * q].detach().numpy()
            f_star = self.prior.model.defunc(0, xu).squeeze()[q : 2 * q].detach().numpy()

            ident = np.block(
                [
                    [np.eye(q), np.eye(q) * self.dt, np.zeros((q, self.u_dim))],
                    [np.zeros((q, q)), np.eye(q), np.zeros((q, self.u_dim))],
                ]
            )
            AB_t = np.block([[D_qqd_qdd * self.dt ** 2], [D_qqd_qdd * self.dt]]) + ident

            c_t = (
                np.hstack([f_star * self.dt ** 2, f_star * self.dt])
                - mean_traj[t][: self.x_dim]
                + mean_traj[t][self.x_dim : 2 * self.x_dim]
            )
            c_t[:q] += mean_traj[t][self.x_dim + q : self.x_dim + 2 * q] * self.dt

            self.AB[t] = AB_t
            self.c[t] = c_t
            self.W[t] = 0.1 * np.eye(self.x_dim)  # TODO better noise reprenstation

        return self.AB, self.c, self.W

    def save(self, name):
        f = open(name, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, name):
        f = open(name, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
