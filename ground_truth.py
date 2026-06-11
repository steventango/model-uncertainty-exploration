import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from env_config import make_state


_DUMMY_KEY = jax.random.key(0)


def true_transition(env, params, env_name, s1, s2, action):
    """Ground-truth (delta_obs, reward, terminated) for a single transition."""
    state = make_state(env_name, s1, s2)
    obs = env.get_obs(state)
    if isinstance(env.action_space(params), spaces.Discrete):
        action = jnp.asarray(action, dtype=jnp.int32)
    _, _, reward, terminated, _, info = env.step(_DUMMY_KEY, state, action, params)
    return info["next_obs"] - obs, reward, terminated.astype(jnp.float32)


def true_delta_obs(env, params, env_name, s1, s2, action):
    """Batched ground-truth observation deltas, shape (N, obs_dim)."""
    delta_obs, _, _ = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return delta_obs


def true_reward(env, params, env_name, s1, s2, action):
    """Batched ground-truth rewards, shape (N,)."""
    _, reward, _ = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return reward


def true_terminated(env, params, env_name, s1, s2, action):
    """Batched ground-truth termination flags, shape (N,)."""
    _, _, terminated = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return terminated
