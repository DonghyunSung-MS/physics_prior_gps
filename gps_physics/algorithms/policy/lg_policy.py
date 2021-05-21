import numpy as np
from numpy.core.shape_base import hstack

from gps_physics.algorithms.policy.policy import Policy


class LGPolicy(Policy):
    def __init__(self, hyperparams, K=None, k=None, cov=None):
        super().__init__(hyperparams)
        self.x_dim = hyperparams["x_dim"]
        self.u_dim = hyperparams["u_dim"]
        self.dt = hyperparams["dt"]
        self.T = hyperparams["T"]

        self.init_policy(K, k, cov)

    def init_policy(self, K=None, k=None, cov=None):
        if K is None:
            self.K = 0.01 * np.random.randn(self.T, self.u_dim, self.x_dim)
        else:
            assert K.shape[0] == self.T and K.shape[1] == self.u_dim and K.shape[2] == self.x_dim
            self.K = K

        if k is None:
            self.k = 0.01 * np.random.randn(self.T, self.u_dim)
        else:
            assert k.shape[0] == self.T and k.shape[1] == self.u_dim
            self.k = k

        if cov is None:
            self.cov = np.stack([np.eye(self.u_dim) for _ in range(self.T)])
        else:
            assert cov.shape[0] == self.T and cov.shape[1] == self.u_dim and cov.shape[2] == self.u_dim
            self.cov = cov

    def backward(self, mean_traj, dynamics, cost, **kwargs):
        gl_policy = kwargs.get("gl_policy")
        eta = kwargs.get("eta")

        T = mean_traj.shape[0]
        AB, c, W = dynamics.fit(mean_traj)

        x_dim = self.x_dim
        u_dim = self.u_dim
        xu_dim = self.x_dim + self.u_dim

        # intialize matrix for traj opt
        Q_xuxu = np.zeros((T, xu_dim, xu_dim))
        Q_xu = np.zeros((T, xu_dim))

        V_xx = np.zeros((T + 1, x_dim, x_dim))
        V_x = np.zeros((T + 1, x_dim))

        self.K = np.zeros((T, self.u_dim, self.x_dim))
        self.k = np.zeros((T, self.u_dim))
        self.cov = np.zeros((T, self.u_dim, self.u_dim))

        for t in range(T - 1, -1, -1):
            l_xuxu = None
            l_xu = None
            l_c = None

            if gl_policy:
                l_xuxu, l_xu, l_c = cost.get_apporx_cost(mean_traj[t][self.x_dim :], eta, gl_policy)
            else:
                l_xuxu, l_xu, l_c = cost.get_apporx_cost(mean_traj[t][self.x_dim :])

            Q_xuxu[t] = l_xuxu + AB[t].T @ V_xx[t + 1] @ AB[t]
            Q_xu[t] = l_xu + V_x[t + 1] @ AB[t] + c[t] @ V_xx[t + 1] @ AB[t]

            # singular careful

            Q_uu_inv = np.linalg.inv(Q_xuxu[t][x_dim:, x_dim:])

            self.K[t] = -Q_uu_inv @ Q_xuxu[t][x_dim:, :x_dim]
            self.k[t] = -Q_uu_inv @ Q_xu[t][x_dim:]
            self.cov[t] = Q_uu_inv

            IK = np.block([[np.eye(x_dim)], [self.K[t]]])  # [[I],[K]]
            zk = np.hstack([np.zeros((x_dim,)), self.k[t]]).reshape(1, -1).T  # [[0], [k]]

            V_xx[t] = IK.T @ Q_xuxu[t] @ IK
            V_x[t] = zk.T @ Q_xuxu[t] @ IK + Q_xu[t] @ IK

    def forward(self, prev_mean_traj, inital_state, dynamics, cost):
        """
        update nominal trajectory
        """
        AB, c, W = dynamics.AB, dynamics.c, dynamics.W

        T = AB.shape[0]

        x_dim = self.x_dim
        u_dim = self.u_dim
        xu_dim = self.x_dim + self.u_dim

        new_mean_traj = np.zeros((T, x_dim + x_dim + u_dim))
        joint_cov = np.zeros((T, x_dim + u_dim, x_dim + u_dim))

        marginal_state_mean = inital_state
        marginal_state_cov = np.eye(x_dim) * 0.01

        l = 0.0

        for t in range(T):
            x_star_next, x_star, u_star = (
                prev_mean_traj[t][:x_dim],
                prev_mean_traj[t][x_dim : x_dim + x_dim],
                prev_mean_traj[t][-u_dim:],
            )
            joint_mean_t = np.hstack([marginal_state_mean, self.get_action(t, marginal_state_mean, x_star, u_star)])

            l += cost.eval_cost(joint_mean_t)

            joint_cov_t = np.block(
                [
                    [marginal_state_cov, marginal_state_cov @ self.K[t].T],
                    [self.K[t] @ marginal_state_cov, self.K[t] @ marginal_state_cov @ self.K[t].T],
                ]
            )

            next_marginal_state_mean = AB[t] @ (joint_mean_t - np.hstack([x_star, u_star])) + c[t] + x_star_next
            next_marginal_state_cov = AB[t] @ joint_cov_t @ AB[t].T

            new_mean_traj[t] = np.hstack([next_marginal_state_mean, joint_mean_t])
            joint_cov[t] = joint_cov_t

            marginal_state_mean = next_marginal_state_mean
            marginal_state_cov = marginal_state_cov

        return new_mean_traj, joint_cov, l

    def fit(self, intial_state, dynamics, prev_mean_traj, cost, **kwargs):
        self.backward(prev_mean_traj, dynamics, cost, **kwargs)
        mean_traj, joint_cov, l = self.forward(prev_mean_traj, intial_state, dynamics, cost)

        return mean_traj, joint_cov, l

    def get_action(self, t, state, x_star, u_star):
        pre_action = self.K[t] @ (state - x_star) + self.k[t] + u_star
        return pre_action
