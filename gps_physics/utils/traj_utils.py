import numpy as np


def kl_trajectory(mean_traj, lg_policy, gl_policy):
    kl = 0.0
    T = mean_traj.shape[0]
    x_dim = gl_policy.x_dim
    u_dim = gl_policy.u_dim

    for t in range(T):
        state = mean_traj[t][x_dim : x_dim + x_dim]
        action = mean_traj[t][-u_dim:]

        loc_mean = lg_policy.get_action(t, state, state, action)
        loc_cov = lg_policy.cov[t]

        lg_mean = gl_policy.get_action(state)
        lg_cov = gl_policy.pi_cov

        lg_cov_inv = np.linalg.inv(lg_cov)
        mean_diff = loc_mean - lg_mean

        print(f"mean_diff: {mean_diff}")

        kl += (
            np.trace(lg_cov_inv @ loc_cov)
            + (mean_diff.reshape(1, -1) @ lg_cov_inv @ mean_diff)[0]
            - u_dim
            + np.log(np.linalg.det(lg_cov))
            - np.log(np.linalg.det(loc_cov))
        )

    return kl*0.5


def eta_adjust(kl_traj, cur_eta, min_eta, max_eta, epsilon):
    new_eta = None

    if kl_traj < 0.9 * epsilon:
        new_eta = min([0.1 * cur_eta, (max_eta * cur_eta) ** 0.5])
    elif kl_traj >= 0.9 * epsilon and kl_traj <= 1.1 * epsilon:
        new_eta = cur_eta
    else:
        new_eta = max([0.1 * cur_eta, (cur_eta * min_eta) ** 0.5])
    
    return new_eta
