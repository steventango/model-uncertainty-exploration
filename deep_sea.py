import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.bsuite.deep_sea import DeepSea, EnvParams


@struct.dataclass
class DeepSeaParams(EnvParams):
    action_map: jax.Array | None = None


class DeepSeaOneHot(environment.Environment):
    """DeepSea as in the VBE paper: deterministic transitions, a random
    per-cell action map (which action means "right") fixed per seed, tabular
    one-hot obs.

    Reachable cell (row, column), column <= row, is class row * (row + 1) / 2 +
    column; one extra last class marks "past the bottom row", so every
    observation, terminal included, is one-hot.
    """

    discrete_obs = True

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self._env = DeepSea(size)
        rows, cols = jnp.tril_indices(size)
        self.cell_rows, self.cell_cols = rows, cols
        self.cell_map = (
            jnp.full((size, size), -1).at[rows, cols].set(rows * (rows + 1) // 2 + cols)
        )
        self.num_cells = size * (size + 1) // 2
        self.goal = self.num_cells - 1

    def params_for_seed(self, seed) -> DeepSeaParams:
        action_map = jax.random.bernoulli(
            jax.random.key(seed), 0.5, (self.size, self.size)
        ).astype(jnp.float32)
        return DeepSeaParams(max_steps_in_episode=self.size, action_map=action_map)

    @property
    def default_params(self) -> DeepSeaParams:
        return self.params_for_seed(0)

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
        state = state.replace(action_mapping=params.action_map)
        return self._obs(state), state

    def obs_to_reward_terminated(self, obs, action, next_obs, params):
        a = jnp.reshape(action, ()).astype(jnp.float32)
        right_action = params.action_map[self.cell_rows, self.cell_cols]
        right = a * right_action + (1 - a) * (1 - right_action)
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
