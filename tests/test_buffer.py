"""Tests for mue/data buffer."""

import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from mue.data import BufferConfig, build_buffer
from mue.data.buffer import ReplayBuffer


def _build(env_id="Pendulum-v1", buffer_size=50):
    env = gym.make(env_id)
    config = BufferConfig(buffer_size=buffer_size)
    buffer = build_buffer(config, env)
    env.close()
    return buffer


def _add_transition(buffer, value, *, done=False):
    value = float(value)
    buffer.add(
        jnp.full(buffer.obs_dim, value),
        jnp.full(buffer.act_dim, -value),
        jnp.full(buffer.obs_dim, value + 0.5),
        reward=value + 1.0,
        done=done,
    )


def test_build_returns_replay_buffer():
    buffer = _build()
    assert isinstance(buffer, ReplayBuffer)
    assert buffer.max_length == 50
    assert buffer.obs_dim == 3
    assert buffer.act_dim == 1


def test_empty_batch_has_zero_leading_dimension():
    buffer = _build()

    batch = buffer.batch()

    assert len(buffer) == 0
    assert batch.obs.shape == (0, 3)
    assert batch.action.shape == (0, 1)
    assert batch.next_obs.shape == (0, 3)
    assert batch.reward.shape == (0,)
    assert batch.done.shape == (0,)


def test_add_increments_length():
    buffer = _build()
    buffer.add(
        jnp.zeros(3),
        jnp.zeros(1),
        jnp.ones(3),
        1.0,
        False,
    )
    assert len(buffer) == 1


def test_batch_contains_all_values_in_insert_order_before_capacity():
    buffer = _build()
    for value in range(5):
        _add_transition(buffer, value, done=value == 3)

    batch = buffer.batch()

    assert batch.obs.shape == (5, 3)
    assert batch.action.shape == (5, 1)
    assert batch.next_obs.shape == (5, 3)
    assert batch.reward.shape == (5,)
    assert batch.done.shape == (5,)
    np.testing.assert_allclose(batch.obs[:, 0], np.arange(5, dtype=np.float32))
    np.testing.assert_allclose(batch.action[:, 0], -np.arange(5, dtype=np.float32))
    np.testing.assert_allclose(
        batch.next_obs[:, 0], np.arange(5, dtype=np.float32) + 0.5
    )
    np.testing.assert_allclose(batch.reward, np.arange(5, dtype=np.float32) + 1.0)
    np.testing.assert_allclose(batch.done, np.array([0, 0, 0, 1, 0], dtype=np.float32))


def test_batch_returns_oldest_to_newest_after_wraparound():
    buffer = _build(buffer_size=3)
    for value in range(5):
        _add_transition(buffer, value)

    batch = buffer.batch()

    assert len(buffer) == 3
    np.testing.assert_allclose(batch.obs[:, 0], np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_allclose(
        batch.action[:, 0], np.array([-2, -3, -4], dtype=np.float32)
    )
    np.testing.assert_allclose(
        batch.next_obs[:, 0], np.array([2.5, 3.5, 4.5], dtype=np.float32)
    )
    np.testing.assert_allclose(batch.reward, np.array([3, 4, 5], dtype=np.float32))


def test_discrete_action_space_uses_single_action_dim():
    buffer = _build(env_id="CartPole-v1")
    buffer.add(
        jnp.zeros(4),
        jnp.array([1.0]),
        jnp.ones(4),
        1.0,
        False,
    )
    batch = buffer.batch()
    assert batch.obs.shape == (1, 4)
    assert batch.action.shape == (1, 1)
    np.testing.assert_allclose(batch.action, np.array([[1.0]], dtype=np.float32))


def test_scalar_action_is_stored_as_single_action_dimension():
    buffer = _build(env_id="CartPole-v1")

    buffer.add(
        jnp.zeros(4),
        jnp.array(1.0),
        jnp.ones(4),
        1.0,
        True,
    )

    batch = buffer.batch()
    np.testing.assert_allclose(batch.action, np.array([[1.0]], dtype=np.float32))
    np.testing.assert_allclose(batch.done, np.array([1.0], dtype=np.float32))
