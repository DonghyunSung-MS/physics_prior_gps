from cvxpy.expressions.cvxtypes import problem
from cvxpy.reductions.solvers import solver
import numpy as np
from numpy.core.shape_base import hstack
import cvxpy as cp

from gps_physics.algorithms.policy.policy import Policy
import time

class MPCPolicy(Policy):
    def __init__(self, x_dim, u_dim, dt, hyperparams):
        super().__init__(hyperparams)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt = dt
        self.is_first = True
        self.last_u = None

    def forward(self, prev_mean_traj, inital_state, AB, c, cost):
        """
        update nominal trajectory
        """

        T = AB.shape[0]

        x_dim = self.x_dim
        u_dim = self.u_dim
        xu_dim = self.x_dim + self.u_dim

        new_mean_traj = np.zeros((T, x_dim + x_dim + u_dim))

        objective = 0.0
        constraint = []

        cp_x = cp.Variable((x_dim, T + 1))
        cp_u = cp.Variable((u_dim, T))
        s = time.time()
        for t in range(T):
            x_star_next, x_star, u_star = (
                prev_mean_traj[t][:x_dim],
                prev_mean_traj[t][x_dim : x_dim + x_dim],
                prev_mean_traj[t][-u_dim:],
            )
            if t==0:
                constraint += [cp_x[:,0] == inital_state - x_star]
            xu_star = np.hstack([x_star, u_star])
            cp_xu_t = cp.hstack([cp_x[:, t], cp_u[:, t]])

            l_xuxu, l_xu, l_c = cost.get_apporx_cost(xu_star)

            objective += 0.5 * cp.quad_form(cp_xu_t, l_xuxu)
            objective += l_xu @ cp_xu_t
            # objective += l_c

            constraint += [cp_x[:, t+1] == AB[t] @ cp_xu_t + c[t]]
            

        constraint += [cp_u + prev_mean_traj[:, -u_dim:].T <= 2.0]
        constraint += [cp_u + prev_mean_traj[:, -u_dim:].T >= -2.0]
        
        # print(f"construct {time.time() - s: 0.5f} ")
        s = time.time()
        problem = cp.Problem(cp.Minimize(objective), constraint)
        l = problem.solve(solver=cp.ECOS)
        # print(f"solve {time.time() - s: 0.5f} ")

        new_traj = np.zeros_like(prev_mean_traj)
        try:
            self.is_first = False
            new_traj[:, :x_dim] = cp_x.value[:, 1:].T
            new_traj[:, x_dim:x_dim*2] = cp_x.value[:, :-1].T
            new_traj[:, -u_dim:] = cp_u.value.T
            
            new_traj += prev_mean_traj
            self.last_u = cp_u.value + prev_mean_traj[:, -u_dim:].T
        except:
            self.is_first = True
            self.last_u = prev_mean_traj[:, -u_dim:].T
            print("unsolved")
            pass
            
        return new_traj, l, cp_u.value + prev_mean_traj[:, -u_dim:].T

    def fit(self, intial_state, dynamics, prev_mean_traj, cost, **kwargs):
        pass
    
    def get_action(self, t, prev_mean_traj, inital_state, AB, c, cost):
        if self.is_first:
            return np.clip(np.random.randn(self.u_dim), -1.0, 1.0)
        else:
            #recideing horizon
            T = prev_mean_traj.shape[0]
            if t+20 < T:
                prev_mean_traj = prev_mean_traj[t:t+20]
                AB = AB[t:t+20]
                c = c[t:t+20]
            else:
                prev_mean_traj = prev_mean_traj[t:]
                AB = AB[t:]
                c = c[t:] 
            
            return self.forward(prev_mean_traj, inital_state, AB, c, cost)[2][:, 0] #first action