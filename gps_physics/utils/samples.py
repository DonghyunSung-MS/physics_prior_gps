import numpy as np
from collections import deque
import pickle

class TrajectoryBuffer:
    def __init__(self, M: int, N: int, T: int, x_dim: int, u_dim: int):
        # M initial condition, N trajectory, T length, next_state, state, action
        self.traj = np.zeros((M ,N, T, x_dim * 2 + u_dim))
        # self.mean_traj = 0.1 * np.random.randn(M, T, x_dim * 2 + u_dim)
        self.mean_traj = np.zeros((M, T, x_dim * 2 + u_dim))
        self.joint_cov = np.zeros((M, T, x_dim + u_dim, x_dim + u_dim))
        self.x_dim = x_dim
        self.u_dim = u_dim

    def get_traj(self, index):
        # get N trajectories "m th" initial condition
        return self.traj[index]  # N, T, next_state, state, action

    def push_transition(self, m, n, t, state, action, next_state):
        transition = np.hstack([next_state, state, action])
        self.traj[m, n, t] = transition

class SuperviseBuffer:
    def __init__(self, max_len):
        self.state_buffer = deque(maxlen=max_len)
        self.action_buffer = deque(maxlen=max_len)

    def push(self, state, action):
        self.state_buffer.append(state)
        self.action_buffer.append(action)

    def get_data(self):
        return np.stack(self.state_buffer), np.stack(self.action_buffer)

    def save(self, name):
        f = open(name, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, name):
        f = open(name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 

if __name__ =="__main__":
    fname = "test.pkl"

    test = ReplayBuffer(100)
    for _ in range(10):
        test.push(np.ones(1), np.zeros(2))

    print(test.get_data())

    test.save(fname)

    new = ReplayBuffer(10)
    new.load(fname)
    print(new.get_data())
    print(new.state_buffer)
        
