import numpy as np


class Cost:
    def __init__(self):
        self.l_xuxu = np.array(np.nan)
        self.l_xu = np.array(np.nan)
        self.l_c = np.array(np.nan)

class SinglePendulmCost(Cost):
    def __init__(self, angle_w, vel_w, u_w):
        super().__init__()
        self.l_xuxu = np.diag(np.array([angle_w, vel_w, u_w]))
        self.l_xu = lambda xu: self.l_xuxu @ xu

        self.l_c = 0.0

    def get_apporx_cost(self, xu):
        return self.l_xuxu, self.l_xu(xu), self.l_c

    def eval_cost(self, xu):
        cost = 0.5 * np.squeeze(xu.reshape(1, -1) @ self.l_xuxu @ xu.reshape(-1, 1))
        return cost