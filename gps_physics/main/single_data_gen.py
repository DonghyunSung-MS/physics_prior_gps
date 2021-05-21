import argparse
import os
from ast import parse

import numpy as np
import toml
import torch
import torchdyn

from gps_physics.algorithms.cost.cost import SinglePendulmCost
from gps_physics.algorithms.dynamics.lnn_dynamics import DynamicsLRLNN
from gps_physics.algorithms.policy.lg_policy import LGPolicy
from gps_physics.algorithms.policy.mpc_policy import MPCPolicy
from gps_physics.algorithms.policy.NN_policy import NNPolicy
from gps_physics.gym_env.single_pendulm import SinglePendulmEnv
from gps_physics.utils.samples import SuperviseBuffer, TrajectoryBuffer
from gps_physics.utils.traj_utils import *

parser = argparse.ArgumentParser(description="Physics Prior Policy Search")
parser.add_argument("--path", type=str, help="path to config file")
parser.add_argument("--type", type=str, help="Constrained LQR(CLQR) or LQR")
args = parser.parse_args()

with open(args.path) as conffile:
    CONFIG = toml.loads(conffile.read())

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
np.set_printoptions(precision=4)

print(f"{args.type} Guide policy search")


def main():
    MAX_ITER = CONFIG["max_iter"]

    M = CONFIG["M"]  # num initial
    N = CONFIG["N"]  # num traj per initial condition
    T = CONFIG["T"]  # horizon

    # env
    x_dim = CONFIG["x_dim"]
    u_dim = CONFIG["u_dim"]
    max_torque = CONFIG["max_torque"]
    mass, length, dt = CONFIG["mass"], CONFIG["length"], CONFIG["dt"]
    env = SinglePendulmEnv(mass, length, dt, max_torque=max_torque)
    sing_pen_cost = SinglePendulmCost(1.0, 0.1, 0.001)

    dynamics_lr = DynamicsLRLNN(CONFIG)
    mpc_policies = [MPCPolicy(CONFIG, u_min=-max_torque, u_max=max_torque) for _ in range(M)]
    lg_policies = [LGPolicy(CONFIG) for _ in range(M)]

    data_buffer = SuperviseBuffer(int(1e5))

    env.reset()
    reset_states = [np.array([0.1, 0.0]), np.array([np.pi / 2.0 + 0.1, 0.0]), np.array([-np.pi / 2.0, 0.0])]

    returns = np.ones(M) * -np.inf
    prev_returns = np.ones(M) * -np.inf
    learning = True

    exp_buffer = []

    for i in range(MAX_ITER):
        print(f"\n-------------{i}th iteration-------------\n")
        iter_traj = TrajectoryBuffer(M, N, T, x_dim, u_dim)
        mean_traj = None
        if i == 0:
            mean_traj = iter_traj.mean_traj
        else:
            mean_traj = exp_buffer[i - 1].mean_traj
        for m in range(M):
            avg_returns = 0.0
            AB, c = None, None

            if args.type == "CLQR":
                if i != 0:
                    AB, c, W = dynamics_lr.fit(mean_traj[m])
                    mpc_policies[m].forward(mean_traj[m], reset_states[m], AB, c, sing_pen_cost)

            for n in range(N):
                env.reset()
                env.state = reset_states[m]
                obs = env._get_obs()  # x0
                for t in range(T):
                    action = None

                    if args.type == "LQR":
                        action = lg_policies[m].get_action(
                            t, obs, mean_traj[m][t][x_dim : x_dim * 2], mean_traj[m][t][-u_dim:]
                        )
                    elif args.type == "CLQR":
                        action = mpc_policies[m].get_action(t, mean_traj[m], obs, AB, c, sing_pen_cost)

                    next_obs, reward, done, _ = env.step(action)

                    avg_returns += reward

                    iter_traj.push_transition(m, n, t, obs, action, next_obs)
                    if not learning:
                        data_buffer.push(obs, action)

                    # if n == 0:
                    #     env.render()

                    obs = next_obs

            returns[m] = avg_returns / N

        delta = np.max(np.abs(prev_returns - returns))
        print(delta, returns, prev_returns)
        if delta > 0.5:
            # dynamics learning
            exp_buffer.append(iter_traj)
            dynamics_lr.updata_prior(exp_buffer)

            # update trajectory
            if args.type == "LQR":
                for m in range(M):
                    dynamics_lr.fit(mean_traj[m])
                    mean_traj_at_m = None
                    joint_cov_at_m = None
                    mean_traj_at_m, joint_cov_at_m, l = lg_policies[m].fit(
                        reset_states[m], dynamics_lr, mean_traj[m], sing_pen_cost
                    )

                    print(f"update {m} init condition")
                    exp_buffer[i].mean_traj[m] = mean_traj_at_m
            elif args.type == "CLQR":
                for m in range(M):
                    if i == 0:
                        mean_traj[m] = np.mean(iter_traj.traj[m], axis=0)
                    print(f"update {m} init condition")
                    AB, c, W = dynamics_lr.fit(mean_traj[m])
                    new_traj, new_est_cost, _ = mpc_policies[m].forward(mean_traj[m], reset_states[m], AB, c, sing_pen_cost)
                    exp_buffer[i].mean_traj[m] = new_traj

        else:
            print("Pass Learning")
            exp_buffer.append(exp_buffer[i - 1])
            learning = False

        prev_returns = returns.copy()

    data_buffer.save(args.type + "_single_pen_data.pkl")
    dynamics_lr.save(args.type + "_single_pen_dynamics.pkl")


if __name__ == "__main__":
    main()
