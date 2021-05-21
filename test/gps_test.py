import os

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

    dynamics_lr = DynamicsLRLNN(config)

    lg_policy_list = []
    gl_policy = NNPolicy(x_dim, u_dim, hyperparams=config["global_policy"])

    # set initial policy
    for m in range(M):
        lg = LGPolicy(config)
        lg.K = 0.01 * np.random.randn(T, u_dim, x_dim)
        lg.k = 0.01 * np.random.randn(T, u_dim)
        lg.cov = np.stack([np.eye(u_dim) for _ in range(T)])
        lg_policy_list.append(lg)

    # sing_pen_augcost = SinglePendulmAugCost(1.0, 0.1, 0.001)
    sing_pen_cost = SinglePendulmCost(1.0, 0.1, 0.001)

    env.reset()

    reset_states = [np.array([0.1, 0.0]), np.array([np.pi / 2.0 + 0.1, 0.0]), np.array([-np.pi / 2.0, 0.0])]

    exp_buffer = []

    for i in range(MAX_ITER):
        iter_traj = TrajectoryBuffer(M, N, T, x_dim, u_dim)
        mean_traj = None
        if i == 0:
            mean_traj = iter_traj.mean_traj
        else:
            mean_traj = exp_buffer[i - 1].mean_traj
        for m in range(M):
            for n in range(N):
                # print(m, n)
                env.reset()
                env.state = reset_states[m]
                obs = env._get_obs()  # x0
                for t in range(T):
                    action = None
                    # if i==0:
                    action = lg_policy_list[m].get_action(
                        t, obs, mean_traj[m][t][x_dim : x_dim * 2], mean_traj[m][t][-u_dim:], max_torque
                    )
                    if action[0] >= 2.0 or action[0] <= -2.0:
                        print(f"init {m}, time {t}, act: {action[0]:0.1f}")
                    next_obs, reward, done, _ = env.step(action[0])
                    iter_traj.push_transition(m, n, t, obs, action, next_obs)
                    if n == 0:
                        env.render()

                    obs = next_obs

        # dynamics learning
        exp_buffer.append(iter_traj)
        dynamics_lr.updata_prior(exp_buffer)

        # if i == 0:
        #     lg_K = []
        #     lg_k = []
        #     lg_cov = []
        #     for m in range(M):
        #         lg_K.append(lg_policy_list[m].K)
        #         lg_k.append(lg_policy_list[m].k)
        #         lg_cov.append(lg_policy_list[m].cov)
        #         iter_traj.mean_traj[m] = np.mean(iter_traj.traj[m], axis=0)

        #     gl_policy.fit(np.stack(lg_K), np.stack(lg_k), np.stack(lg_cov), iter_traj)

        # lg_K = []
        # lg_k = []
        # lg_cov = []

        for m in range(M):
            epsilon = 1.0
            eta = 1.0
            dynamics_lr.fit(mean_traj[m])
            mean_traj_at_m = None
            joint_cov_at_m = None
            mean_traj_at_m, joint_cov_at_m, l = lg_policy_list[m].fit(
                reset_states[m], dynamics_lr, mean_traj[m], sing_pen_cost
            )
            # gps iteration
            # for _ in range(lg_step):
            #     mean_traj_at_m, joint_cov_at_m, l = lg_policy_list[m].fit(
            #         reset_states[m], dynamics_lr, mean_traj[m], sing_pen_augcost, gl_policy=gl_policy, eta=eta
            #     )
            #     kl_traj = kl_trajectory(mean_traj_at_m, lg_policy_list[m], gl_policy)
            #     eta = eta_adjust(kl_traj, eta, eta_min, eta_max, epsilon)
            #     print(f"{i}th iter eta: {eta}, kl_traj {kl_traj}")

            print(f"{m}th init cost: {l}")
            exp_buffer[i].mean_traj[m] = mean_traj_at_m
        #     iter_traj.joint_cov[m] = joint_cov_at_m

        #     lg_K.append(lg_policy_list[m].K)
        #     lg_k.append(lg_policy_list[m].k)
        #     lg_cov.append(lg_policy_list[m].cov)

        # gl_policy.fit(np.stack(lg_K), np.stack(lg_k), np.stack(lg_cov), iter_traj)
