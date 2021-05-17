from gps_physics.algorithms.policy.policy import Policy
from gps_physics.utils.ptu import mlp
import numpy as np
import torch
from torch.autograd.functional import jacobian
import torch.utils.data as pt_data

class NNPolicy(Policy):
    def __init__(self, x_dim, u_dim, hyperparams):
        super().__init__(hyperparams)
        self.x_dim = x_dim
        self.u_dim = u_dim

        self.model = mlp(x_dim, hyperparams["hidden_size"], u_dim, hyperparams["layer_depth"], hyperparams["act"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=hyperparams["lr"])

        self.pi_cov = np.eye(u_dim)
        
    def to_lg_policy(self, mean_trajectory):
        T = mean_trajectory.shape[0]
        x_dim = self.x_dim
        u_dim  = self.u_dim

        self.K = np.zeros((T, u_dim, x_dim))
        self.k = np.zeros((T, u_dim))
        self.cov = np.zeros((T, u_dim, u_dim))

        for t in range(T):
            state = mean_trajectory[t][self.x_dim : self.x_dim*2]
            action = mean_trajectory[t][self.x_dim*2 : ]
            pt_state = torch.FloatTensor(state)

            self.K[t] = jacobian(self.model, pt_state).detach().numpy()
            self.k[t] = self.model(pt_state)
            self.cov = self.pi_cov

        return self.K, self.k, self.cov
            
    def fit(self, lg_K, lg_k, lg_cov, traj_buffer):
        M, N, T, _ = traj_buffer.traj.shape
        pt_state = torch.FloatTensor(traj_buffer.traj[:, :, :, self.x_dim: 2*self.x_dim]).reshape(-1, self.x_dim)

        mean_action = np.squeeze(lg_K @ traj_buffer.mean_traj[:, :, self.x_dim: 2*self.x_dim, np.newaxis] + lg_k[:,:,:, np.newaxis], -1) # M x T x U
        pt_mean_action = torch.FloatTensor(mean_action).unsqueeze(1).repeat(1, N, 1, 1) # M x N x T x U
        pt_mean_action = pt_mean_action.reshape(-1, self.u_dim)

        pt_cov = torch.FloatTensor(lg_cov).unsqueeze(1).repeat(1, N, 1, 1, 1) # M x N x T x U x U
        pt_cov = pt_cov.reshape(-1, self.u_dim, self.u_dim)
        
        train_data = pt_data.TensorDataset(pt_state, pt_mean_action, pt_cov)
        train_dataloader = pt_data.DataLoader(train_data, batch_size=30, shuffle=True)

        epoch = self._hyperparams["epoch_per_iteration"]

        for i in range(epoch):
            for idx, data in enumerate(train_dataloader):
                state_sample, mean_acion, pt_cov = train_data

                state_sample = state_sample.requires_grad_(True)
                mean_acion = mean_acion.requires_grad_(True)
                pt_cov = pt_cov.requires_grad_(True)

                action_sample = self.model(pt_state)

                loss = (action_sample - mean_acion).unsqueeze(1) @ pt_cov.inverse() @ (action_sample - mean_acion).unsqueeze(2)
                loss = loss.mean()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def get_action(self, state):
        pt_state = torch.FloatTensor(state)
        action = self.model(pt_state).detach().numpy()
        return action

        