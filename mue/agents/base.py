from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp

@dataclass
class BaseAgentConfig:
    update_steps: int = 5


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
