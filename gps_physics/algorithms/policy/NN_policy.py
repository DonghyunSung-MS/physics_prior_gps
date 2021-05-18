import numpy as np
import torch
import torch.utils.data as pt_data
from torch.autograd.functional import jacobian

from gps_physics.algorithms.policy.policy import Policy
from gps_physics.utils.ptu import mlp


class NNPolicy(Policy):
    def __init__(self, x_dim, u_dim, hyperparams):
        super().__init__(hyperparams)
        self.x_dim = x_dim
        self.u_dim = u_dim

        self.model = mlp(x_dim, hyperparams["hidden_size"], u_dim, hyperparams["layer_depth"], hyperparams["act"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=hyperparams["lr"])

        self.pi_cov = np.eye(u_dim)

    def to_lg_policy(self, xu):
        x_dim = self.x_dim
        u_dim = self.u_dim

        K = np.zeros((u_dim, x_dim))
        k = np.zeros((u_dim))
        cov = np.zeros((u_dim, u_dim))

        state = xu[: self.x_dim]
        action = xu[self.x_dim :]

        pt_state = torch.FloatTensor(state)

        K = jacobian(self.model, pt_state).detach().numpy()
        k = self.model(pt_state).detach().numpy() - K @ state + action
        cov = self.pi_cov

        return K, k, cov

    def fit(self, lg_K, lg_k, lg_cov, traj_buffer):
        M, N, T, _ = traj_buffer.traj.shape

        np_state_sample = traj_buffer.traj[:, :, :, self.x_dim : 2 * self.x_dim]  # M N T U
        np_state_nominal = traj_buffer.mean_traj[:, np.newaxis, :, self.x_dim : 2 * self.x_dim]  # M 1 T X
        np_action_nominal = traj_buffer.mean_traj[:, np.newaxis, :, -self.u_dim :]  # M 1 T U

        pt_state = torch.FloatTensor(np_state_sample).reshape(-1, self.x_dim)
        lg_K_tile = np.tile(lg_K[:, np.newaxis, :, :, :], (1, N, 1, 1, 1))
        mean_action = (
            np.squeeze(lg_K_tile @ (np_state_sample - np_state_nominal)[:, :, :, :, np.newaxis], axis=-1)
            + lg_k[:, np.newaxis, :, :]
            + np_action_nominal
        )  # M x N x T x U

        pt_mean_action = torch.FloatTensor(mean_action)  # M x N x T x U
        pt_mean_action = pt_mean_action.reshape(-1, self.u_dim)
        print(pt_mean_action.shape)

        pt_cov = torch.FloatTensor(lg_cov).unsqueeze(1).repeat(1, N, 1, 1, 1)  # M x N x T x U x U
        pt_cov = pt_cov.reshape(-1, self.u_dim, self.u_dim)

        train_data = pt_data.TensorDataset(pt_state, pt_mean_action, pt_cov)
        train_dataloader = pt_data.DataLoader(train_data, batch_size=self._hyperparams["batch_size"], shuffle=True)

        epoch = self._hyperparams["epoch_per_iteration"]
        log_loss = []
        for i in range(epoch):
            for idx, data in enumerate(train_dataloader):
                state_sample, acion_sample, pt_cov_sample = data

                state_sample = state_sample.requires_grad_(True)
                acion_sample = acion_sample.requires_grad_(True)
                pt_cov_sample = pt_cov_sample.requires_grad_(True)

                action_pred = self.model(state_sample)
                loss = (
                    (action_pred - acion_sample).unsqueeze(1)
                    @ pt_cov_sample.inverse()
                    @ (action_pred - acion_sample).unsqueeze(2)
                )
                loss = loss.mean()

                log_loss.append(loss.detach().item())
                # if idx % 100 == 0:
                #     print(f"policy loss {loss}")
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        print(f"policy loss : {np.mean(log_loss)}")

    def get_action(self, state):
        pt_state = torch.FloatTensor(state)
        action = self.model(pt_state).detach().numpy()
        return action
