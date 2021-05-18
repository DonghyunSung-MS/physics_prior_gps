import torch
import torchdyn
import wandb
import numpy as np

class LNNLearner:
    def __init__(self, model: torchdyn.models.NeuralDE, dt, x_dim, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.dt = dt
        self.x_dim = x_dim
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        if batch is None:
            print(batch)
        x, y = batch
        y_hat = self.model.defunc(0, x)  # static training: we do not solve the ODE
        # y_hat = self.model.defunc.m.discrete_predict(x)[:, :self.x_dim]
        loss = self.loss(y_hat, y)
        # self.log('train_loss', loss)
        return loss

    def update(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def loss(self, y_hat, y):  # predict, target, mseloss
        return ((y - y_hat) ** 2).mean()

    def fit(self, epoch, dataloader):
        log_loss = []
        for i in range(epoch):
            for idx, data in enumerate(dataloader):
                loss = self.training_step(data, idx)
                self.update(loss)
                log_loss.append(loss.detach().item())
                # if idx % 1000 == 0:
                #     print(f"dyn loss : {loss}")

        print(f"dyn loss : {np.mean(log_loss)}")