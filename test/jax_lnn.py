import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit, random, vmap
from jax.experimental import optimizers, stax
from jax.experimental.ode import odeint
from jax.random import PRNGKey


def normalize_coord(state):
    q, qdot = jnp.split(state, 2)
    q_normalized = (q + np.pi) % (2 * np.pi) - np.pi
    return jnp.concatenate([q_normalized, qdot])


def jax_hessian(f):
    # we are using double jacfwd.
    # jacfwd is more efficient for 'tall' jac matrices.
    # jacrev is more efficient for 'wide' jac matrices.
    # in our case, input is mostly tall, not wide.
    # so jacfwd(jacfwd(x)) is faster than jacfwd(jacrev)
    return jit(jacfwd(jacfwd(f)))


# def jax_hessian_diag(f, input):
#     H = jax_hessian(f)(input)
#     diag = H.diagonal(0, 1, 2)
#     return diag

# vjax_hessian_diag = vmap(jax_hessian_diag, in_axes=(None, 0))


def lagrangian_forward_dynamics(L, state, action):
    D = state.shape[0] // 2

    state = normalize_coord(state)
    q, qdot = jnp.split(state, 2)

    # L_H = jax_hessian(L)(state)
    # L_jac= jax.jacobian(L)(state)

    # L_H_q_dot_inv = jnp.linalg.pinv(L_H[D:, D:]) # M ^ -1
    # L_q = L_jac[:D] # grad L w.r.t q
    # L_q_q_dot = L_H[:D, D:] # hess(q,qdot)

    # q_ddot = L_H_q_dot_inv @ (L_q - L_q_q_dot @ qdot + action)
    return None


# init_random_params, mlp = stax.serial(
#                                         stax.Dense(128),
#                                         stax.Softplus,
#                                         stax.Dense(128),
#                                         stax.Softplus,
#                                         stax.Dense(1),
#                                     )


rng = jax.random.PRNGKey(0)
# in_shape = (-1, 4)
# outshape, net_params = init_random_params(rng, in_shape)

state = jax.random.normal(rng, (4,))
action = jax.random.normal(rng, (2,))

device_


L = lambda x: x.T @ x
print(state, L)

import time

s = time.time()
lagrangian_forward_dynamics(L, state, action)
print(time.time() - s)

s = time.time()
for i in range(10):
    lagrangian_forward_dynamics(L, state, action)
print(time.time() - s)
