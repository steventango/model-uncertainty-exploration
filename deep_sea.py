import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.bsuite.deep_sea import DeepSea, EnvParams


class OneHot(spaces.Space):
    def __init__(self, n: int):
        self.n = n
        self.shape = (n,)
        self.dtype = jnp.float32
        self.low = jnp.zeros(n, jnp.float32)
        self.high = jnp.ones(n, jnp.float32)

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.nn.one_hot(jax.random.randint(key, (), 0, self.n), self.n)

    def contains(self, x: jax.Array) -> jax.Array:
        return (x.shape == self.shape) & (x.sum() == 1) & jnp.all((x == 0) | (x == 1))


@struct.dataclass
class DeepSeaParams(EnvParams):
    seed: int = 0


class DeepSeaOneHot(environment.Environment):
    """DeepSea as in the VBE paper: deterministic transitions, a random
    per-cell action map (which action means "right") fixed per seed, tabular
    one-hot obs.

    Reachable cell (row, column), column <= row, is class row * (row + 1) / 2 +
    column. As in VBE, the observation after the bottom row is the reset cell 0;
    termination is read from the current cell being on the bottom row.
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
        return DeepSeaParams(max_steps_in_episode=self.size, seed=seed)

    def action_map(self, params):
        return jax.random.bernoulli(
            jax.random.key(params.seed), 0.5, (self.size, self.size)
        ).astype(jnp.float32)

    @property
    def default_params(self) -> DeepSeaParams:
        return self.params_for_seed(0)

    def _obs(self, state):
        cell = jnp.where(
            state.row < self.size,
            self.cell_map[jnp.minimum(state.row, self.size - 1), state.column],
            0,
        )
        return jax.nn.one_hot(cell, self.num_cells)

    def step_env(self, key, state, action, params):
        _, state, reward, done, info = self._env.step_env(key, state, action, params)
        return self._obs(state), state, reward, done, info

    def reset_env(self, key, params):
        _, state = self._env.reset_env(key, params)
        state = state.replace(action_mapping=self.action_map(params))
        return self._obs(state), state

    def obs_to_reward_terminated(self, obs, action, next_obs, params):
        a = jnp.reshape(action, ()).astype(jnp.float32)
        right_action = self.action_map(params)[self.cell_rows, self.cell_cols]
        right = a * right_action + (1 - a) * (1 - right_action)
        p_right = (obs * right).sum()
        reward = p_right * (obs[self.goal] - 0.01 / self.size)
        return reward.astype(jnp.float32), obs[self.num_cells - self.size :].sum() > 0.5

    def observation_space(self, params):
        return OneHot(self.num_cells)

    def action_space(self, params=None):
        return spaces.Discrete(2)

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return f"DeepSea-{self.size}"
