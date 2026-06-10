import jax
import jax.numpy as jnp
from gymnax.environments.classic_control.pendulum import EnvState


_DUMMY_KEY = jax.random.key(0)


def _make_state(theta, theta_dot):
    return EnvState(theta=theta, theta_dot=theta_dot, last_u=jnp.zeros(()), time=0)


def true_transition(env, params, theta, theta_dot, action):
    """Ground-truth (delta_obs, reward) for a single transition."""
    state = _make_state(theta, theta_dot)
    obs = env.get_obs(state)
    next_obs, _, reward, _, _ = env.step_env(_DUMMY_KEY, state, action, params)
    return next_obs - obs, reward


def true_delta_obs(env, params, theta, theta_dot, action):
    """Batched ground-truth observation deltas, shape (N, obs_dim)."""
    delta_obs, _ = jax.vmap(lambda th, td, a: true_transition(env, params, th, td, a))(
        theta, theta_dot, action
    )
    return delta_obs


def true_reward(env, params, theta, theta_dot, action):
    """Batched ground-truth rewards, shape (N,)."""
    _, reward = jax.vmap(lambda th, td, a: true_transition(env, params, th, td, a))(
        theta, theta_dot, action
    )
    return reward
