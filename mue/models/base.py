from typing import Protocol

import jax.numpy as jnp

from mue.types import ModelPrediction


class WorldModel(Protocol):
    def fit(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        next_obs: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> dict[str, float]:
        """Train on dataset. Returns loss metrics."""
        ...

    def predict(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> ModelPrediction:
        """Predict next state distribution."""
        ...
