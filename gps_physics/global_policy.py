import argparse

import numpy as np
import toml

from gps_physics.algorithms.policy.NN_policy import NNPolicy
from gps_physics.gym_env.single_pendulm import SinglePendulmEnv
from gps_physics.utils.samples import SuperviseBuffer

parser = argparse.ArgumentParser(description="Physics Prior Policy Search N.N")
parser.add_argument("--path", type=str, help="path to config file")
parser.add_argument("--dpkl", type=str, help="path to data pkl file")
parser.add_argument("--ppkl", type=str, help="path to policy pkl file")
parser.add_argument("--train", action="store_true", help="train or not")
args = parser.parse_args()

with open(args.path) as conffile:
    CONFIG = toml.loads(conffile.read())

global_policy = NNPolicy(CONFIG)

if args.train:
    buffer = SuperviseBuffer(int(1e5))
    buffer.load(args.pkl)

    states, actions = buffer.get_data()
    global_policy.fit(states, actions)

else:
    global_policy.load(args.ppkl)
    num_test = 10

    T = CONFIG["T"]  # horizon
    max_torque = CONFIG["max_torque"]
    mass, length, dt = CONFIG["mass"], CONFIG["length"], CONFIG["dt"]
    env = SinglePendulmEnv(mass, length, dt, max_torque=max_torque)

    env.reset()
    for _ in range(num_test):
        reset_state = np.random.uniform(low=np.array([-np.pi, -0.01]), high=np.array([np.pi, 0.01]))
        env.state = reset_state
        obs = env._get_obs()

        for t in range(T):
            action = global_policy.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
