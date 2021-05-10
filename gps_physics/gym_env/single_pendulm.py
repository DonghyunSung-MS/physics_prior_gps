import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import odeint

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def eom(x, t, m, g, l, u):
    q, q_dot = x
    q = angle_normalize(q)
    return np.array([q_dot, u /(m * (l**2)) + g / l*np.sin(q)])

class SinglePendulmEnv(gym.Env):
    """[summary]
    referece: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    simple single pendulum

    point mass equation of motion

    ml^2 thet_tt - mglsin(theta) = Q, where downward theta = pi upward = 0.0
    
    """
    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 30
        }

    def __init__(self, m, l, dt, g=9.8):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.viewer = None

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] 

    def _get_obs(self):
        return self.state

    def step(self, u):
        th, thdot = self.state  # th := theta
        t = np.linspace(0, self.dt, int(self.dt*1000)) # continous -> time step 0.001
        sol = odeint(eom, self.state, t, args=(self.m, self.g, self.l, u))
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        self.state = sol[-1]
        self.last_u = u
        self.state[0] = angle_normalize(self.state[0])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([np.pi, 0.0])
        self.last_u = None
        return self._get_obs()

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1, .02)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            mass = rendering.make_circle(0.1)
            mass.set_color(.3, .3, .8)
            self.mass_transform = rendering.Transform()
            mass.add_attr(self.mass_transform)
            self.viewer.add_geom(mass)

            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        self.mass_transform.set_translation(np.cos(self.state[0] + np.pi / 2), np.sin(self.state[0] + np.pi / 2))
        
        self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__=="__main__":
    m, l, dt = 1.0, 0.25, 0.01

    env = SinglePendulmEnv(m, l, dt)
    env.reset()
    for i in range(1000):
        obs, reward, done, info = env.step(0.02)
        print(obs)
        print()
        env.render()