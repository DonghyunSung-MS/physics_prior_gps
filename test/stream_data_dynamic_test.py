import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
import torchdyn

import gps_physics
from gps_physics.algorithms.cost.cost import SinglePendulmCost
from gps_physics.algorithms.dynamics.lnn_dynamics import DynamicsLRLNN
from gps_physics.algorithms.policy.lg_policy import LGPolicy
from gps_physics.gym_env.single_pendulm import SinglePendulmEnv
from gps_physics.utils.samples import TrajectoryBuffer

matplotlib.use("TkAgg")

config_path = os.path.join(gps_physics.ROOT_DIR, "algorithms/config/gps_phsics.toml")
with open(config_path) as conffile:
    config = toml.loads(conffile.read())

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])


if __name__ == "__main__":

    MAX_ITER = config["gps"]["max_iter"]
    M = config["gps"]["M"]  # num initial
    N = config["gps"]["N"]  # num traj per initial condition
    T = config["gps"]["T"]  # horizon

    mass, length, dt = 1.0, 0.25, 0.01
    env = SinglePendulmEnv(mass, length, dt)

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]

    dynamics_lr = DynamicsLRLNN(x_dim, u_dim, dt, config)

    lg_policy_list = []

    # set initial policy
    for m in range(M):
        lg = LGPolicy(x_dim, u_dim, dt, config)
        lg.K = np.zeros((T, u_dim, x_dim))
        lg.k = np.random.randn(T, u_dim) * 1.5
        lg.cov = np.stack([np.eye(u_dim) for _ in range(T)])
        lg_policy_list.append(lg)

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
                    action = lg_policy_list[m].get_action(t, obs, mean_traj[m][t][x_dim : x_dim * 2], mean_traj[m][t][-u_dim:])
                    next_obs, reward, done, _ = env.step(action[0])
                    iter_traj.push_transition(m, n, t, obs, action, next_obs)
                    if n == 0:
                        env.render()

                    obs = next_obs

        # dynamics learning
        exp_buffer.append(iter_traj)
        dynamics_lr.updata_prior(exp_buffer)

        for m in range(M):
            dynamics_lr.fit(mean_traj[m])
            mean_traj_at_m, joint_cov_at_m, l = lg_policy_list[m].fit(
                reset_states[m], dynamics_lr, mean_traj[m], sing_pen_cost
            )
            print(f"{m}th init cost: {l}")
            iter_traj.mean_traj[m] = mean_traj_at_m
            iter_traj.joint_cov[m] = joint_cov_at_m

        print("it's done")
        # compare dynamics
        # u = np.random.randn()*1.5
        # x = reset_states[2]
        # time = (0.0, 1.0)
        # space = torch.linspace(*time, 1000)
        # pt_xu = torch.FloatTensor(np.hstack([x, u]).reshape(1, -1))

        # def angle_normalize(x):
        #     return ((x + np.pi) % (2 * np.pi)) - np.pi

        # def eom(t, xu):
        #     q, q_dot, u = torch.split(xu, 1, 1)
        #     q = angle_normalize(q)
        #     b = 0.0
        #     q_ddot = (u - b * q_dot) / (mass * length ** 2) + 9.8 / length * torch.sin(q)
        #     return torch.cat([q_dot, q_ddot, torch.zeros_like(u)], dim=1)

        # gt_traj = torchdyn.odeint(eom, pt_xu, space, method="rk4").squeeze().detach()
        # print("torchdyn done")

        # traj = (
        #     dynamics_lr.prior.model.trajectory(pt_xu, space)
        #     .squeeze()
        #     .detach()
        # )
        # print("torchdyn done")
        # print(gt_traj.shape)
        # plt.plot(space, gt_traj[:, 0])
        # plt.plot(space, gt_traj[:, 1])
        # plt.plot(space, traj[:, 0], "--")
        # plt.plot(space, traj[:, 1], "--")
        # plt.legend(["gt_pos", "gt_vel"] + ["pr_pos", "pr_vel"])
        # # plt.legend()
        # plt.show()
