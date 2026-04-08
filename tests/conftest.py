"""Shared fixtures for the MUE test suite."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from mue.config import ExperimentConfig
from mue.types import ModelPrediction  # used by DummyModel

OBS_DIM = 3
ACT_DIM = 1
BATCH_SIZE = 8


@pytest.fixture
def rng_key():
    return jax.random.key(42)


@pytest.fixture
def pendulum_env():
    env = gym.make("Pendulum-v1")
    yield env
    env.close()


@pytest.fixture
def small_config():
    return ExperimentConfig(
        env_id="Pendulum-v1",
        seed=42,
        max_steps=5,
        eval_episodes=1,
    )


@pytest.fixture
def synthetic_transitions(rng_key):
    """Generate N random transitions for Pendulum-like env."""
    N = 20
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    obs = jax.random.normal(k1, (N, OBS_DIM))
    actions = jax.random.uniform(k2, (N, ACT_DIM), minval=-2.0, maxval=2.0)
    next_obs = obs + 0.01 * jax.random.normal(k3, (N, OBS_DIM))
    rewards = jax.random.normal(k4, (N,))
    dones = jnp.zeros(N)
    return obs, actions, next_obs, rewards, dones


class DummyModel:
    """Trivial WorldModel returning identity predictions with small noise."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        self.obs_dim = obs_dim
        self.out_dim = obs_dim + 1

    def fit(self, obs, actions, next_obs, rewards):
        return {"model/loss": 0.0}

    def predict(self, obs, actions):
        batch = obs.shape[0]
        mean = jnp.concatenate([obs, jnp.zeros((batch, 1))], axis=-1)
        cov = 0.01 * jnp.broadcast_to(
            jnp.eye(self.out_dim), (batch, self.out_dim, self.out_dim)
        )
        return ModelPrediction(mean=mean, covariance=cov)


@pytest.fixture
def dummy_model():
    return DummyModel()
