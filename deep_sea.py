import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from gymnax.environments.bsuite.deep_sea import DeepSea, EnvParams


class DeepSeaOneHot(environment.Environment):
    """DeepSea as in the VBE paper: deterministic transitions, a fixed random
    per-cell action map (which action means "right"), tabular one-hot obs.

    Reachable cell (row, column), column <= row, is class row * (row + 1) / 2 +
    column; one extra last class marks "past the bottom row", so every
    observation, terminal included, is one-hot.
    """

    discrete_obs = True

    def __init__(self, size: int, map_seed: int = 0):
        super().__init__()
        self.size = size
        self._env = DeepSea(size)
        self.action_map = jax.random.bernoulli(
            jax.random.key(map_seed), 0.5, (size, size)
        ).astype(jnp.float32)
        rows, cols = jnp.tril_indices(size)
        self.cell_map = (
            jnp.full((size, size), -1).at[rows, cols].set(rows * (rows + 1) // 2 + cols)
        )
        self.num_cells = size * (size + 1) // 2
        self.goal = self.num_cells - 1
        self.right_action = (
            jnp.zeros(self.num_cells)
            .at[self.cell_map[rows, cols]]
            .set(self.action_map[rows, cols])
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.size)

    def _obs(self, state):
        cell = jnp.where(
            state.row < self.size,
            self.cell_map[jnp.minimum(state.row, self.size - 1), state.column],
            self.num_cells,
        )
        return jax.nn.one_hot(cell, self.num_cells + 1)

    def step_env(self, key, state, action, params):
        _, state, reward, done, info = self._env.step_env(key, state, action, params)
        return self._obs(state), state, reward, done, info

    def reset_env(self, key, params):
        _, state = self._env.reset_env(key, params)
        state = state.replace(action_mapping=self.action_map)
        return self._obs(state), state

    def obs_to_reward_terminated(self, obs, action, next_obs):
        a = jnp.reshape(action, ()).astype(jnp.float32)
        right = a * self.right_action + (1 - a) * (1 - self.right_action)
        p_right = (obs[:-1] * right).sum()
        reward = p_right * (obs[self.goal] - 0.01 / self.size)
        return reward.astype(jnp.float32), next_obs[-1] > 0.5

    def observation_space(self, params):
        return spaces.Box(0.0, 1.0, (self.num_cells + 1,), jnp.float32)

    def action_space(self, params=None):
        return spaces.Discrete(2)

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return f"DeepSea-{self.size}"
