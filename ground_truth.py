import jax
import jax.numpy as jnp

from env_config import make_state


_DUMMY_KEY = jax.random.key(0)


def true_transition(env, params, env_name, s1, s2, action):
    """Ground-truth (delta_obs, reward) for a single transition."""
    state = make_state(env_name, s1, s2)
    obs = env.get_obs(state)
    next_obs, _, reward, _, _ = env.step_env(_DUMMY_KEY, state, action, params)
    return next_obs - obs, reward


def true_delta_obs(env, params, env_name, s1, s2, action):
    """Batched ground-truth observation deltas, shape (N, obs_dim)."""
    delta_obs, _ = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return delta_obs


def true_reward(env, params, env_name, s1, s2, action):
    """Batched ground-truth rewards, shape (N,)."""
    _, reward = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return reward
