from typing import NamedTuple

import jax.numpy as jnp


class ModelPrediction(NamedTuple):
    mean: jnp.ndarray  # (batch, obs_dim + 1) — last dim is reward
    covariance: jnp.ndarray  # (batch, obs_dim + 1, obs_dim + 1) — total predictive
    epistemic_covariance: jnp.ndarray | None = None  # reducible uncertainty
    aleatoric_covariance: jnp.ndarray | None = None  # irreducible noise
