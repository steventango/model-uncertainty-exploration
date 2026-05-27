from typing import Protocol

import jax.numpy as jnp


class Agent(Protocol):
    def act(self, obs: jnp.ndarray) -> jnp.ndarray:
        ...

    def update(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_obs: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> dict[str, float]:
        ...
