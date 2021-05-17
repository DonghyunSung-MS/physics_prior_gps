import numpy as np


class TrajectoryBuffer:
    def __init__(self, M: int, N: int, T: int, x_dim: int, u_dim: int):
        # M initial condition, N trajectory, T length, next_state, state, action
        self.traj = np.zeros((M, N, T, x_dim * 2 + u_dim))
        self.mean_traj = np.zeros((M, T, x_dim * 2 + u_dim))
        self.joint_cov = np.zeros((M, T, x_dim + u_dim, x_dim + u_dim))

    def get_traj(self, index):
        # get N trajectories "m th" initial condition
        return self.traj[index]  # N, T, next_state, state, action

    def push_transition(self, m, n, t, state, action, next_state):
        transition = np.hstack([next_state, state, action])
        self.traj[m, n, t] = transition
