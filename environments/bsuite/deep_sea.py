import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.bsuite.deep_sea import DeepSea, EnvParams


@struct.dataclass
class DeepSeaParams(EnvParams):
    action_mapping_key: jax.Array = struct.field(
        default_factory=lambda: jax.random.key(0)
    )


class KeyedDeepSea(DeepSea):
    """gymnax DeepSea as in the VBE paper: deterministic transitions, a random
    per-cell action map (which action means "right") fixed per seed, and a
    tabular observation.

    Upstream draws the action map once in ``__init__``, so it cannot be vmapped
    over seeds; here it is drawn from ``params.action_mapping_key``. Upstream's
    observation is a (size, size) grid; here it is the integer index of the
    reachable cell (row, column), column <= row: ``row * (row + 1) / 2 + column``.
    As in VBE, the observation after the bottom row is the reset cell 0.
    """

    def __init__(self, size: int):
        super().__init__(size)
        rows, cols = jnp.tril_indices(size)
        self.cell_rows, self.cell_cols = rows, cols
        self.cell_map = (
            jnp.full((size, size), -1).at[rows, cols].set(rows * (rows + 1) // 2 + cols)
        )
        self.num_cells = size * (size + 1) // 2
        self.goal = self.num_cells - 1

    def params_for_seed(self, seed) -> DeepSeaParams:
        return DeepSeaParams(
            max_steps_in_episode=self.size, action_mapping_key=jax.random.key(seed)
        )

    @property
    def default_params(self) -> DeepSeaParams:
        return self.params_for_seed(0)

    def action_map(self, params: DeepSeaParams) -> jax.Array:
        return jax.random.bernoulli(
            params.action_mapping_key, 0.5, (self.size, self.size)
        ).astype(jnp.int32)

    def reset_env(self, key, params):
        obs, state = super().reset_env(key, params)
        return obs, state.replace(action_mapping=self.action_map(params))

    def get_obs(self, state, params=None, key=None):
        on_grid = state.row < self.size
        cell = self.cell_map[jnp.minimum(state.row, self.size - 1), state.column]
        return jnp.where(on_grid, cell, 0).astype(jnp.int32)

    def observation_space(self, params) -> spaces.Discrete:
        return spaces.Discrete(self.num_cells)

    def reward_table(self, action, params) -> jax.Array:
        """r(cell, action) for every cell, shape (num_cells,).

        With terminated_table, the model env's oracle (see
        OneHotObservationWrapper.obs_to_reward_terminated).
        """
        right_action = self.action_map(params)[self.cell_rows, self.cell_cols]
        right = right_action == jnp.reshape(action, ()).astype(jnp.int32)
        goal = jax.nn.one_hot(self.goal, self.num_cells)
        return right * (goal - params.unscaled_move_cost / self.size)

    def terminated_table(self) -> jax.Array:
        """Whether each cell is on the bottom row, shape (num_cells,)."""
        return jnp.arange(self.num_cells) >= self.num_cells - self.size
