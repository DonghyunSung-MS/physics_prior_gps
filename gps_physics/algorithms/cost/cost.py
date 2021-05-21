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


class DoublePendulmCost(Cost):
    def __init__(self, angle_w, vel_w, u_w):
        super().__init__()
        self.l_xuxu = np.diag(np.array([angle_w, angle_w, vel_w, vel_w, u_w]))
        self.l_xu = lambda xu: self.l_xuxu @ xu

        self.l_c = 0.0

    def get_apporx_cost(self, xu):
        return self.l_xuxu, self.l_xu(xu), self.l_c

    def eval_cost(self, xu):
        cost = 0.5 * np.squeeze(xu.reshape(1, -1) @ self.l_xuxu @ xu.reshape(-1, 1))
        return cost


class SinglePendulmAugCost(SinglePendulmCost):
    def __init__(self, angle_w, vel_w, u_w):
        super().__init__(angle_w, vel_w, u_w)

    def get_apporx_cost(self, xu, eta, nn_policy):
        K, k, cov = nn_policy.to_lg_policy(xu)
        cov_inv = np.linalg.inv(cov)

        policy_quad = np.block([[K.T @ cov_inv @ K, -K.T @ cov_inv], [-cov_inv @ K, cov_inv]])

        policy_linear = np.hstack([(k.reshape(1, -1) @ cov_inv @ K).reshape(-1), -(k.reshape(1, -1) @ cov_inv).reshape(-1)])

        policy_const = (k.reshape(1, -1) @ cov_inv @ k)[0]

        return self.l_xuxu / eta + policy_quad, self.l_xu(xu) / eta + policy_linear, self.l_c / eta + policy_const

    def eval_cost(self, xu):
        cost = 0.5 * np.squeeze(xu.reshape(1, -1) @ self.l_xuxu @ xu.reshape(-1, 1))
        return cost
