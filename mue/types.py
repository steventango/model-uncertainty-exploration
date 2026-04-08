from typing import NamedTuple

import jax.numpy as jnp


class Transition(NamedTuple):
    obs: jnp.ndarray  # (obs_dim,)
    action: jnp.ndarray  # (act_dim,)
    next_obs: jnp.ndarray  # (obs_dim,)
    reward: jnp.ndarray  # scalar
    done: jnp.ndarray  # scalar


class ModelPrediction(NamedTuple):
    mean: jnp.ndarray  # (batch, obs_dim + 1) — last dim is reward
    covariance: jnp.ndarray  # (batch, obs_dim + 1, obs_dim + 1) — total predictive
    epistemic_covariance: jnp.ndarray | None = None  # reducible uncertainty
    aleatoric_covariance: jnp.ndarray | None = None  # irreducible noise


class BatchTransition(NamedTuple):
    obs: jnp.ndarray  # (batch, obs_dim)
    action: jnp.ndarray  # (batch, act_dim)
    next_obs: jnp.ndarray  # (batch, obs_dim)
    reward: jnp.ndarray  # (batch,)
    done: jnp.ndarray  # (batch,)
