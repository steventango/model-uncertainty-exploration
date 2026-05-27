import copy

import gymnasium as gym
import jax.numpy as jnp

from mue.agents.base import Agent


class RandomAgent(Agent):
    def __init__(self, action_space: gym.Space, seed: int):
        self.action_space = copy.deepcopy(action_space)
        self.seed = int(seed)
        self.action_space.seed(self.seed)

    def act(self, obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.action_space.sample())

    def update(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_obs: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> dict[str, float]:
        return {}
