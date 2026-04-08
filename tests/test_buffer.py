"""Tests for mue/data buffer."""

import gymnasium as gym
import jax
import jax.numpy as jnp

from mue.data import BufferConfig, build_buffer


def _build(env_id="Pendulum-v1", buffer_size=50, batch_size=4):
    env = gym.make(env_id)
    config = BufferConfig(buffer_size=buffer_size, batch_size=batch_size)
    buffer, state = build_buffer(config, env)
    env.close()
    return buffer, state


def _timestep(obs_dim=3, act_dim=1, val=0.0):
    return {
        "obs": jnp.full((1, obs_dim), val),
        "action": jnp.full((1, act_dim), val),
        "reward": jnp.array([val]),
        "done": jnp.array([0.0]),
    }


def test_init():
    _, state = _build()
    assert state.current_index == 0


def test_add():
    buffer, state = _build()
    state = buffer.add(state, _timestep())
    assert state.current_index == 1


def test_can_sample_after_adds():
    buffer, state = _build()
    state = buffer.add(state, _timestep(val=0.0))
    state = buffer.add(state, _timestep(val=1.0))
    assert buffer.can_sample(state)


def test_sample_shapes():
    buffer, state = _build(batch_size=4)
    for i in range(10):
        state = buffer.add(state, _timestep(val=float(i)))
    batch = buffer.sample(state, jax.random.key(0))
    first = batch.experience.first
    second = batch.experience.second
    assert first["obs"].shape == (4, 3)
    assert first["action"].shape == (4, 1)
    assert first["reward"].shape == (4,)
    assert second["obs"].shape == (4, 3)


def test_consecutive_pairs():
    buffer, state = _build(batch_size=2)
    for i in range(10):
        state = buffer.add(
            state,
            {
                "obs": jnp.array([[float(i)]]),
                "action": jnp.array([[0.0]]),
                "reward": jnp.array([0.0]),
                "done": jnp.array([0.0]),
            },
        )
    batch = buffer.sample(state, jax.random.key(0))
    first_obs = batch.experience.first["obs"]
    second_obs = batch.experience.second["obs"]
    assert jnp.allclose(second_obs - first_obs, 1.0)


def test_discrete_action_space():
    buffer, state = _build(env_id="CartPole-v1")
    obs_dim = 4  # CartPole obs
    state = buffer.add(
        state,
        {
            "obs": jnp.zeros((1, obs_dim)),
            "action": jnp.array([[1.0]]),
            "reward": jnp.array([1.0]),
            "done": jnp.array([0.0]),
        },
    )
    state = buffer.add(
        state,
        {
            "obs": jnp.ones((1, obs_dim)),
            "action": jnp.array([[0.0]]),
            "reward": jnp.array([1.0]),
            "done": jnp.array([0.0]),
        },
    )
    assert buffer.can_sample(state)
    batch = buffer.sample(state, jax.random.key(0))
    assert batch.experience.first["obs"].shape == (4, 4)
    assert batch.experience.first["action"].shape == (4, 1)
