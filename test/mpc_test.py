import os
from cvxpy.settings import NONNEG

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
import torchdyn

import gps_physics
from gps_physics.algorithms.cost.cost import SinglePendulmAugCost, SinglePendulmCost
from gps_physics.algorithms.dynamics.lnn_dynamics import DynamicsLRLNN
from gps_physics.algorithms.policy.lg_policy import LGPolicy
from gps_physics.algorithms.policy.NN_policy import NNPolicy
from gps_physics.algorithms.policy.mpc_policy import MPCPolicy

from gps_physics.gym_env.single_pendulm import SinglePendulmEnv
from gps_physics.utils.samples import TrajectoryBuffer
from gps_physics.utils.traj_utils import *

matplotlib.use("TkAgg")

config_path = os.path.join(gps_physics.ROOT_DIR, "algorithms/config/gps_phsics.toml")
with open(config_path) as conffile:
    config = toml.loads(conffile.read())

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])


if __name__ == "__main__":
    max_torque = 1.0
    MAX_ITER = config["gps_algo"]["max_iter"]
    M = config["gps_algo"]["M"]  # num initial
    N = config["gps_algo"]["N"]  # num traj per initial condition
    T = config["gps_algo"]["T"]  # horizon

    # lag_multiplier
    eta_min = config["gps_algo"]["eta_min"]
    eta_max = config["gps_algo"]["eta_max"]

    lg_step = config["gps_algo"]["lg_step"]

    mass, length, dt = 0.5, 0.25, 0.01
    env = SinglePendulmEnv(mass, length, dt)

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]

    dynamics_lr = DynamicsLRLNN(x_dim, u_dim, dt, config)
    mpc_policies = [MPCPolicy(x_dim, u_dim, dt, config) for _ in range(M)]
    sing_pen_cost = SinglePendulmCost(1.0, 0.1, 0.001)


    env.reset()

    reset_states = [np.array([0.1, 0.0]), np.array([np.pi-0.1, 0.0]), np.array([np.pi / 2.0, 0.0])]

    exp_buffer = []

    for i in range(MAX_ITER):
        iter_traj = TrajectoryBuffer(M, N, T, x_dim, u_dim)
        mean_traj = None
        if i == 0:
            mean_traj = iter_traj.mean_traj
        else:
            mean_traj = exp_buffer[i - 1].mean_traj
        for m in range(M):
            AB, c= None, None
            if i!=0:
                AB, c, W = dynamics_lr.fit(mean_traj[m])
                mpc_policies[m].forward(mean_traj[m], reset_states[m], AB, c, sing_pen_cost)

            for n in range(N):
                # print(m, n)
                env.reset()
                env.state = reset_states[m]
                obs = env._get_obs()  # x0
                for t in range(T):
                    action = mpc_policies[m].get_action(t, mean_traj[m], obs, AB, c, sing_pen_cost)
                    if action[0]>=2.0 or action[0]<=-2.0:
                        print(f"init {m}, time {t}, act: {action[0]:0.1f}")
                    next_obs, reward, done, _ = env.step(action[0])
                    iter_traj.push_transition(m, n, t, obs, action, next_obs)
                    if n == 0:
                        env.render()

                    obs = next_obs

        # dynamics learning
        exp_buffer.append(iter_traj)
        dynamics_lr.updata_prior(exp_buffer)

        # update trajectory
        for m in range(M):
            if i == 0:
                mean_traj[m] = np.mean(iter_traj.traj[m], axis=0)
            print(f"update {m} init condition")
            AB, c, W = dynamics_lr.fit(mean_traj[m])
            new_traj, new_est_cost, _ = mpc_policies[m].forward(mean_traj[m], reset_states[m], AB, c, sing_pen_cost)
            exp_buffer[i].mean_traj[m] = new_traj


