import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as pt_data
from torch.autograd.functional import jacobian

from gps_physics.algorithms.policy.policy import Policy
from gps_physics.utils.ptu import mlp, CosSin


class NNPolicy(Policy):
    def __init__(self, hyperparams):
        self.x_dim = hyperparams["x_dim"]
        self.u_dim = hyperparams["u_dim"]

        hyperparams = hyperparams["global_policy"]
        super().__init__(hyperparams)
        self.model = torch.nn.Sequential(
                            *[
                                nn.Linear(x_dim, hyperparams["hidden_size"]),
                                nn.Softplus(),
                                nn.Linear(hyperparams["hidden_size"], hyperparams["hidden_size"]),
                                nn.Softplus(),
                                nn.Linear(hyperparams["hidden_size"], u_dim),
                                nn.Tanh(),
                                # nn.Linear(x_dim, u_dim),
                            ]
                        )
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

        K = jacobian(self.model, pt_state.reshape(1, -1)).reshape(u_dim, x_dim).detach().numpy()
        k = self.model(pt_state.reshape(1, -1)).reshape(-1).detach().numpy() - K @ state
        cov = self.pi_cov

        return K, k, cov

    def fit(self, state:np.array, action:np.array):
        
        pt_state = torch.FloatTensor(state)
        pt_action = torch.FloatTensor(action)
        
        train_data = pt_data.TensorDataset(pt_state, pt_action)
        train_dataloader = pt_data.DataLoader(train_data, batch_size=self._hyperparams["batch_size"], shuffle=True)

        epoch = self._hyperparams["epoch_per_iteration"]
        log_loss = []
        for i in range(epoch):
            for idx, data in enumerate(train_dataloader):
                state_sample, action_sample = data

                state_sample = state_sample.requires_grad_(True)
                action_sample = action_sample.requires_grad_(True)

                action_pred = self.model(state_sample)
                loss = (action_sample - action_pred)**2
                loss = loss.mean()

                log_loss.append(loss.detach().item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        print(f"policy loss : {np.mean(log_loss)}")

    def get_action(self, state):
        pt_state = torch.FloatTensor(state)
        action = self.model(pt_state.reshape(1, -1)).reshape(-1).detach().numpy()
        return action

if __name__ == "__main__":
    from torch.autograd.functional import jacobian
    x_dim = 2
    u_dim = 1

    model = torch.nn.Sequential(
                            *[
                                CosSin(x_dim, angular_dims=[0]),
                                nn.Linear(x_dim + len([0]), 128),
                                nn.Softplus(),
                                nn.Linear(128, 128),
                                nn.Softplus(),
                                nn.Linear(128, u_dim),
                            ]
                        )

    print(jacobian(model, torch.ones(2).reshape(1,-1)).reshape(-1))