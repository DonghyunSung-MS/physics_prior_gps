import warnings
from os import path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.integrate import odeint


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def eom(x, t, m, g, l, u):
    """Equation of Motion double pendulum

    coordinate we choose: all q are zeros -> upright each q are global coordinate instead of local difference.
    this is somewhat differece from robotics convention, but it does not change the system(symmetries of physics).

    Args:
        x (numpy.array): state of the system x = [q; dq], shape (nq + nv, )
        t (float): time index for scipy.odeint
        m (numpy.array): state of the system x = [q; dq], shape (nq + nv, )
        g (float): gravity
        l (numpy.array): length of system, shape (nq, )
        u (numpy.array): input force(torque), shape (nq, )

    Returns:
        [numpy.array]: return time derivative of state, shape (nq, )
    """
    q, q_dot = np.split(x, 2)
    q = angle_normalize(q)

    q1, q2 = q
    q1_dot, q2_dot = q_dot

    m1, m2 = m
    l1, l2 = l
    u1, u2 = u

    alpha1 = m2 * l2 / (m1 + m2) / l1 * np.cos(q1 - q2)
    alpha2 = l1 / l2 * np.cos(q1 - q2)

    f1 = u1 / (m1 + m2) / l1 ** 2 + g / l1 * np.sin(q1) - (m2 / (m1 + m2)) * l2 / l1 * q2_dot ** 2 * np.sin(q1 - q2)
    f2 = u2 / (m1 * l2 ** 2) + l1 / l2 * q1_dot ** 2 * np.sin(q1 - q2) + g / l2 * np.sin(q2)

    q_ddot = np.array([f1 - alpha1 * f2, -alpha2 * f1 + f2]) / (1 - alpha1 * alpha2)

    x_dot = np.hstack([q_dot, q_ddot])

    return x_dot


class DoublePendulmEnv(gym.Env):  # Acrobat underactuated
    """Double Pendulum Environment

    referece:https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
    simple double pendulum
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, m, l, dt, g=9.8, max_torque=1.0):
        self.max_speed = 8
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.last_u = None
        self.mass_ratio = m[1] / m[0]

        self.input_matrix = np.array([0.0, 1.0])
        self.viewer = None

        high = np.array([np.pi, np.pi, self.max_speed, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return self.state

    def step(self, u):
        if isinstance(u, np.ndarray) and u.shape[0] == 1:
            u = u[0]
        if u > self.max_torque + 0.1 or u < -self.max_torque - 0.1:
            warnings.warn(f"action {u:0.3f} beyond limit")

        q, q_dot = np.split(self.state, 2)

        t = np.linspace(0, self.dt, int(self.dt * 1000))  # continous -> time step 0.001

        input = self.input_matrix * u
        sol = odeint(eom, self.state, t, args=(self.m, self.g, self.l, input))

        costs = np.sum(angle_normalize(q) ** 2) + 0.1 * np.sum(q_dot ** 2) + 0.001 * (u ** 2)

        self.state = sol[-1]

        self.last_u = u

        self.state[0] = angle_normalize(self.state[0])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.state = np.array([np.pi, np.pi, 0.0, 0.0])
        self.state = np.array([0.0, 0.1, 0.0, 0.0])

        self.last_u = None
        return self._get_obs()

    def render(self, mode="human"):
        box_size = 2.0
        vir_l = self.l / np.sum(self.l) * box_size * 0.8

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-box_size, box_size, -box_size, box_size)

            rod1 = rendering.make_capsule(vir_l[0], 0.02)
            rod1.set_color(0.8, 0.3, 0.3)
            self.pole_transform1 = rendering.Transform()
            rod1.add_attr(self.pole_transform1)
            self.viewer.add_geom(rod1)

            rod2 = rendering.make_capsule(vir_l[1], 0.02)
            rod2.set_color(0.8, 0.3, 0.3)
            self.pole_transform2 = rendering.Transform()
            rod2.add_attr(self.pole_transform2)
            self.viewer.add_geom(rod2)

            mass1 = rendering.make_circle(0.1)
            mass1.set_color(0.3, 0.3, 0.8)
            self.mass_transform1 = rendering.Transform()
            mass1.add_attr(self.mass_transform1)
            self.viewer.add_geom(mass1)

            mass2 = rendering.make_circle(self.mass_ratio ** 0.3 * 0.1)
            mass2.set_color(0.3, 0.3, 0.8)
            self.mass_transform2 = rendering.Transform()
            mass2.add_attr(self.mass_transform2)
            self.viewer.add_geom(mass2)

            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)

        q, _ = np.split(self.state, 2)
        q = q + np.pi / 2.0

        x1 = vir_l[0] * np.cos(q[0])
        x2 = vir_l[0] * np.cos(q[0]) + vir_l[1] * np.cos(q[1])

        y1 = vir_l[0] * np.sin(q[0])
        y2 = vir_l[0] * np.sin(q[0]) + vir_l[1] * np.sin(q[1])

        self.pole_transform1.set_rotation(q[0])

        self.pole_transform2.set_rotation(q[1])
        self.pole_transform2.set_translation(x1, y1)

        self.mass_transform1.set_translation(x1, y1)
        self.mass_transform2.set_translation(x2, y2)

        self.imgtrans.set_translation(x1, y1)

        self.imgtrans.set_scale(-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    m, l, dt = np.array([0.1, 0.1]), np.array([1.0, 1.0]), 0.01
    g = 9.8
    t = 0.0
    x = np.random.randn(4)
    u = np.random.randn(2)
    # print(eom(x, t, m, g, l, u))
    env = DoublePendulmEnv(m, l, dt)
    env.reset()
    for i in range(10000):
        obs, reward, done, info = env.step(0.0)
        #     print(obs)
        #     print()
        env.render()
