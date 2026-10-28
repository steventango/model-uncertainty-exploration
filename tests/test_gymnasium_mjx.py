"""Gymnasium-on-MJX envs against gymnasium.make(...) on MuJoCo C."""

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from environments import make_state_reconstruction_wrapper
from gymnasium_mjx import GYMNASIUM_MJX_ENVS, make_gymnasium_mjx_env

ENV_NAMES = list(GYMNASIUM_MJX_ENVS)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_matches_gymnasium_float64(env_name):
    """Same start state + same actions -> same obs/reward as Gymnasium (x64).

    half_cheetah's ground contacts amplify float64 rounding ~10x per 3 steps
    (1e-15 at step 0, 1e-5 by step 40), so it is only compared for 20 steps.
    """
    num_steps = 20 if env_name == "HalfCheetah-v5" else 100
    with jax.enable_x64(True):
        env = make_gymnasium_mjx_env(env_name)
        gym_env = gymnasium.make(env_name).unwrapped
        gym_env.reset(seed=0)
        assert env.observation_space(None).shape == gym_env.observation_space.shape
        assert env.action_space(None).shape == gym_env.action_space.shape
        assert env.default_params.max_steps_in_episode == gymnasium.spec(env_name).max_episode_steps

        _, state = env.reset(jax.random.PRNGKey(0))
        gym_env.set_state(np.asarray(state.data.qpos), np.asarray(state.data.qvel))
        np.testing.assert_allclose(env._get_obs(state.data), gym_env._get_obs(), atol=1e-12)

        step = jax.jit(env.step_env)
        rng = np.random.default_rng(0)
        for _ in range(num_steps):
            action = rng.uniform(-1, 1, gym_env.action_space.shape)
            obs, state, reward, terminated, _ = step(None, state, jnp.asarray(action))
            gym_obs, gym_reward, gym_terminated, _, _ = gym_env.step(action)
            np.testing.assert_allclose(obs, gym_obs, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(reward, gym_reward, rtol=1e-7, atol=1e-7)
            assert bool(terminated) == gym_terminated


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_reset_distribution(env_name):
    """Reset qpos/qvel noise has Gymnasium's support and spread."""
    env = make_gymnasium_mjx_env(env_name)
    gym_env = gymnasium.make(env_name).unwrapped
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    _, states = jax.vmap(env.reset)(keys)
    qpos, qvel = np.asarray(states.data.qpos), np.asarray(states.data.qvel)
    gym_qpos, gym_qvel = [], []
    for seed in range(2000):
        gym_env.reset(seed=seed)
        gym_qpos.append(gym_env.data.qpos.copy())
        gym_qvel.append(gym_env.data.qvel.copy())
    for ours, theirs in [(qpos, np.array(gym_qpos)), (qvel, np.array(gym_qvel))]:
        np.testing.assert_allclose(ours.mean(0), theirs.mean(0), atol=0.01)
        np.testing.assert_allclose(ours.std(0), theirs.std(0), rtol=0.1, atol=1e-4)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_vmapped_step_truncates_and_autoresets(env_name):
    env = make_gymnasium_mjx_env(env_name)
    T = env.default_params.max_steps_in_episode
    num_envs = 2
    _, state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    action = jnp.zeros((num_envs,) + env.action_space(None).shape)

    @jax.jit
    def rollout(state):
        def body(state, key):
            keys = jax.random.split(key, num_envs)
            obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(
                keys, state, action
            )
            return state, (obs, reward, terminated, truncated, info["next_obs"])

        return jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(1), T + 1))

    state, (obs, reward, terminated, truncated, next_obs) = rollout(state)
    assert obs.dtype == jnp.float32 and reward.dtype == jnp.float32
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(reward))
    assert not terminated.any()
    assert truncated[T - 1].all() and not truncated[: T - 1].any() and not truncated[T].any()
    # On truncation the returned obs is the fresh reset obs, not next_obs.
    assert not np.allclose(obs[T - 1], next_obs[T - 1])
    np.testing.assert_array_equal(state.time, 1)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_oracle_reward_matches_env_reward(env_name):
    """The obs-based oracle used when --predict_reward_terminated is off."""
    env = make_gymnasium_mjx_env(env_name)
    oracle = make_state_reconstruction_wrapper(env, env_name)
    obs, state = env.reset(jax.random.PRNGKey(0))
    action = jax.random.uniform(jax.random.PRNGKey(1), env.action_space(None).shape, minval=-1, maxval=1)
    next_obs, _, reward, terminated, _ = env.step_env(None, state, action)
    oracle_reward, oracle_terminated = oracle.obs_to_reward_terminated(obs, action, next_obs)
    assert bool(oracle_terminated) == bool(terminated)
    # Reacher's oracle is exact; cheetah/swimmer use end-of-step qvel[0] for
    # the finite-difference x velocity, so only roughly agree.
    tol = 1e-5 if env_name == "Reacher-v5" else 0.5
    np.testing.assert_allclose(oracle_reward, reward, atol=tol)
