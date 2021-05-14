import os

import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from scipy.integrate import odeint

import gps_physics
from gps_physics.algorithms.dynamics.lnn_dynamics import DynamicsLRLNN
from gps_physics.gym_env.single_pendulm import SinglePendulmEnv, eom
from gps_physics.utils.samples import Trajectory

config_path = os.path.join(gps_physics.ROOT_DIR, "algorithms/config/gps_phsics.toml")
with open(config_path) as conffile:
    config = toml.loads(conffile.read())

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])


def policy(state):
    return 0.1 * np.random.randn()
    # return 10 * (0 - state[0]) - state[1]


if __name__ == "__main__":

    MAX_ITER = config["gps"]["max_iter"]
    M = config["gps"]["M"]  # num initial
    N = config["gps"]["N"]  # num traj per initial condition
    T = config["gps"]["T"]  # horizon

    m, l, dt = 1, 0.25, 0.01
    env = SinglePendulmEnv(m, l, dt)

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]

    dynamics_lr = DynamicsLRLNN(x_dim, u_dim, dt, config)

    env.reset()

    reset_states = [np.array([0.1, 0.0]), np.array([np.pi / 2.0, 0.0]), np.array([-np.pi / 2.0, 0.0])]

    traj_buffer = []

    for i in range(MAX_ITER):
        iter_traj = Trajectory(M, N, T, x_dim, u_dim)
        for m in range(M):
            for n in range(N):
                # print(m, n)
                env.reset()
                env.state = reset_states[m]
                obs = env._get_obs()  # x0
                for t in range(T):
                    action = policy(obs)
                    next_obs, reward, done, _ = env.step(action)

                    iter_traj.push_transition(m, n, t, obs, action, next_obs)
                    # env.render()

                    obs = next_obs

        traj_buffer.append(iter_traj.traj)
        dynamics_lr.updata_prior(traj_buffer)
        print("it's done")

        # compare dynamics
        u = 1
        x = reset_states[2]
        space = np.linspace(0, 1, 100)
        sol = odeint(eom, x, space, args=(m, 9.8, l, u))
        print("scipy done")

        traj = (
            dynamics_lr.prior.model.trajectory(torch.FloatTensor(np.hstack([x, u]).reshape(1, -1)), torch.FloatTensor(space))
            .squeeze()
            .detach()
        )
        print("torchdyn done")
        plt.plot(space, sol[:, 0])
        plt.plot(space, sol[:, 1])

        plt.plot(space, traj[:, 0], "--")
        plt.plot(space, traj[:, 1], "--")
        plt.legend(["gt_pos", "gt_vel"] + ["pr_pos", "pr_vel"])
        # plt.legend()
        plt.show()
        # dynamics_lr.fit()
