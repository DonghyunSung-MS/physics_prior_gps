from operator import le
import osqp
import numpy as np
from numpy.core.shape_base import hstack
from scipy import sparse
from gps_physics.algorithms.policy.policy import Policy
import time

class MPCPolicy(Policy):
    def __init__(self, hyperparams, x_min=None, x_max=None, u_min=None, u_max=None):
        super().__init__(hyperparams)
        self.x_dim = hyperparams["x_dim"]
        self.u_dim = hyperparams["u_dim"]
        self.dt = hyperparams["dt"]

        self.is_first = True
        self.last_u = None

        self.x_min = x_min if x_min is not None else np.ones(self.x_dim)*-np.inf
        self.x_max = x_max if x_max is not None else np.ones(self.x_dim)*np.inf

        self.u_min = u_min if u_min is not None else np.ones(self.u_dim)*-np.inf
        self.u_max = u_max if u_max is not None else np.ones(self.u_dim)*np.inf

    def forward(self, prev_mean_traj, inital_state, AB, c, cost):
        """
        update nominal trajectory


        """

        T = AB.shape[0]

        x_dim = self.x_dim
        u_dim = self.u_dim
        xu_dim = self.x_dim + self.u_dim

        new_mean_traj = np.zeros((T, x_dim + x_dim + u_dim))

        #objective
        # X = ($x(0), ... ,$x(T), $u(0), ..., $u(T-1))
        # 0.5 X.T P X + q X
        # s.t -$x_{t+1} + A$x_t + B$u_t + c_t == 0
        
        qp_dim = x_dim * (T + 1) + u_dim * T
        # P = np.zeros((qp_dim, qp_dim))
        # q = np.zeros(qp_dim)
        l_xxs = []
        l_xus = []
        l_uxs = []
        l_uus = []

        l_xs = []
        l_us = []

        A = []
        B = []
        c_ts = []

        prev_state_vector = [] #x*(0), ... ,x*(T)
        prev_action_vector = [] #u*(0), ..., u*(T-1))
        


        for t in range(T):
            x_star_next, x_star, u_star = (
                prev_mean_traj[t][:x_dim],
                prev_mean_traj[t][x_dim : x_dim + x_dim],
                prev_mean_traj[t][-u_dim:],
            )
            prev_state_vector.append(x_star)
            prev_action_vector.append(u_star)
            
            xu_star = np.hstack([x_star, u_star])

            l_xuxu, l_xu, l_c = cost.get_apporx_cost(xu_star)

            l_xxs.append(l_xuxu[:x_dim, :x_dim])
            l_xus.append(l_xuxu[:x_dim, x_dim:])
            l_uxs.append(l_xuxu[x_dim:, :x_dim])
            l_uus.append(l_xuxu[x_dim:, x_dim:])

            l_xs.append(l_xu[:x_dim])
            l_us.append(l_xu[x_dim:])

            A.append(AB[t][:, :x_dim])
            B.append(AB[t][:, x_dim:])
            c_ts.append(c[t])


            if t==T-1:
                prev_state_vector.append(x_star_next)

        prev_traj_vector = np.hstack(prev_state_vector + prev_action_vector)

        #final reward
        l_xxs.append(np.zeros((x_dim, x_dim)))
        l_xs.append(np.zeros(x_dim))
       
        l_xu_block = sparse.vstack([sparse.block_diag(l_xus), sparse.csc_matrix((x_dim, T*u_dim))])
        l_ux_block = sparse.hstack([sparse.block_diag(l_uxs), sparse.csc_matrix((u_dim*T, x_dim))])
        

        P = sparse.vstack([
                            sparse.hstack([sparse.block_diag(l_xxs), l_xu_block]),
                            sparse.hstack([l_ux_block, sparse.block_diag(l_uus)])
                          ], format="csc")
        
        q = np.hstack(l_xs + l_us)

        #constraint
        #linearized dynamics
        Ax = sparse.kron(sparse.eye(T+1), -sparse.eye(x_dim)) + sparse.hstack([sparse.vstack([sparse.csc_matrix((x_dim, T*x_dim)), sparse.block_diag(A)]),
                                                                               sparse.csc_matrix(((T+1)*x_dim, x_dim))])             
        Bu = sparse.vstack([sparse.csc_matrix((x_dim, T*u_dim)), sparse.block_diag(B)])
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-inital_state + prev_traj_vector[:x_dim], -np.hstack(c_ts)])
        ueq = leq

        Aineq = sparse.eye((T+1)*x_dim + T*u_dim)
        lineq = np.hstack([np.kron(np.ones(T + 1), self.x_min), np.kron(np.ones(T), self.u_min)]) - prev_traj_vector
        uineq = np.hstack([np.kron(np.ones(T + 1), self.x_max), np.kron(np.ones(T), self.u_max)]) - prev_traj_vector

        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        res = prob.solve()
        
        x_value = res.x[:x_dim*(T+1)].reshape(-1, x_dim) # T+1 x nx
        u_value = res.x[x_dim*(T+1):].reshape(-1, u_dim) # T x nu

        new_traj = np.zeros_like(prev_mean_traj)

        self.is_first = False
        new_traj[:, :x_dim] = x_value[1:, :]
        new_traj[:, x_dim:x_dim*2] = x_value[:-1,:]
        new_traj[:, -u_dim:] = u_value
        
        new_traj += prev_mean_traj

        return new_traj, l, u_value[0] + prev_mean_traj[0, -u_dim:]

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
            
            return self.forward(prev_mean_traj, inital_state, AB, c, cost)[2] #first action