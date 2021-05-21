import numpy as np


def kl_trajectory(mean_traj, lg_policy, gl_policy):
    kl = 0.0
    T = mean_traj.shape[0]
    x_dim = gl_policy.x_dim
    u_dim = gl_policy.u_dim

    for t in range(T):
        state = mean_traj[t][x_dim : x_dim + x_dim]
        action = mean_traj[t][-u_dim:]

        loc_mean = lg_policy.get_action(t, state, state, action, 1.0)
        loc_cov = lg_policy.cov[t]

        gl_mean = gl_policy.get_action(state)
        gl_cov = gl_policy.pi_cov

        gl_cov_inv = np.linalg.inv(gl_cov)
        mean_diff = loc_mean - gl_mean

        # print(f"mean_diff: {mean_diff}")

        kl += (
            np.trace(gl_cov_inv @ loc_cov)
            + (mean_diff.reshape(1, -1) @ gl_cov_inv @ mean_diff)[0]
            - u_dim
            + np.log(np.linalg.det(gl_cov_inv))
            - np.log(np.linalg.det(loc_cov))
        )

    return kl * 0.5


def eta_adjust(kl_traj, cur_eta, min_eta, max_eta, epsilon):

    new_eta = np.clip(cur_eta + 0.01 * (kl_traj - epsilon), min_eta, max_eta)

    return new_eta
