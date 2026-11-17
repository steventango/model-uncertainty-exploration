import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.bsuite.deep_sea import DeepSea as GymnaxDeepSea
from gymnax.environments.bsuite.deep_sea import EnvParams as GymnaxDeepSeaEnvParams
from gymnax.environments.bsuite.deep_sea import EnvState


@struct.dataclass
class EnvParams(GymnaxDeepSeaEnvParams):
    action_mapping_key: jax.Array = struct.field(
        default_factory=lambda: jax.random.key(0)
    )


class DeepSea(GymnaxDeepSea):
    def __init__(self, size: int):
        super().__init__(size)
        rows, cols = jnp.tril_indices(size)
        self.state_rows, self.state_columns = rows, cols
        self.state_map = (
            jnp.full((size, size), -1).at[rows, cols].set(rows * (rows + 1) // 2 + cols)
        )
        self.n_states = size * (size + 1) // 2

    def params_for_seed(self, seed) -> EnvParams:
        return EnvParams(
            max_steps_in_episode=self.size, action_mapping_key=jax.random.key(seed)
        )

    @property
    def default_params(self) -> EnvParams:
        return self.params_for_seed(0)

    def action_map(self, params: EnvParams) -> jax.Array:
        return jax.random.bernoulli(
            params.action_mapping_key, 0.5, (self.size, self.size)
        ).astype(jnp.int32)

    def reset_env(self, key, params):
        obs, state = super().reset_env(key, params)
        return obs, state.replace(action_mapping=self.action_map(params))

    def get_obs(self, state, params=None, key=None):
        state_index = self.state_map[
            jnp.minimum(state.row, self.size - 1), state.column
        ]
        return jnp.where(self.is_terminated(state, params), 0, state_index).astype(
            jnp.int32
        )

    def observation_space(self, params) -> spaces.Discrete:
        return spaces.Discrete(self.n_states)

    def get_state(self, obs, last_action, time, next_obs=None, params=None):
        return EnvState(
            row=self.state_rows[obs],
            column=self.state_columns[obs],
            bad_episode=False,
            total_bad_episodes=0,
            denoised_return=0,
            optimal_return=0.0,
            action_mapping=self.action_map(params),
            time=time,
        )
