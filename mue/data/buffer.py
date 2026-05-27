from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class ReplayBatch:
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class ReplayBuffer:
    def __init__(
        self,
        max_length: int,
        obs_dim: int,
        act_dim: int,
    ):
        self.max_length = int(max_length)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self._obs = np.zeros((self.max_length, self.obs_dim), dtype=np.float32)
        self._action = np.zeros((self.max_length, self.act_dim), dtype=np.float32)
        self._next_obs = np.zeros((self.max_length, self.obs_dim), dtype=np.float32)
        self._reward = np.zeros((self.max_length,), dtype=np.float32)
        self._done = np.zeros((self.max_length,), dtype=np.float32)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        idx = self._ptr
        self._obs[idx] = np.asarray(obs, dtype=np.float32)
        self._action[idx] = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        self._next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self._reward[idx] = float(reward)
        self._done[idx] = float(done)

        self._ptr = (self._ptr + 1) % self.max_length
        self._size = min(self._size + 1, self.max_length)

    def _ordered_indices(self) -> np.ndarray | slice:
        if self._size < self.max_length or self._ptr == 0:
            return slice(0, self._size)
        return np.concatenate(
            [np.arange(self._ptr, self.max_length), np.arange(0, self._ptr)]
        )

    def batch(self) -> ReplayBatch:
        idx = self._ordered_indices()
        return ReplayBatch(
            obs=jnp.asarray(self._obs[idx]),
            action=jnp.asarray(self._action[idx]),
            next_obs=jnp.asarray(self._next_obs[idx]),
            reward=jnp.asarray(self._reward[idx]),
            done=jnp.asarray(self._done[idx]),
        )
