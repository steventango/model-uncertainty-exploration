from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


_P0, _P_SLOPE, _E_CONST_WH, _HOURS = 3.3, 44.9, 571.6, 12.0
_EP_LEN = 14.0


@struct.dataclass
class PlantEnvState(environment.EnvState):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    time: int


@struct.dataclass
class PlantEnvParams:
    area_min: float
    area_max: float
    act_low: jnp.ndarray
    act_high: jnp.ndarray
    init_areas: jnp.ndarray
    max_steps_in_episode: int


class PlantEnv(environment.Environment[PlantEnvState, PlantEnvParams]):
    """Offline plant-growth environment.

    This env does not simulate its own dynamics: ``step_env`` replays the
    ``next_obs`` already stored on the state rather than computing a transition.
    It is therefore only meaningful when driven by a ``ModelEnvironment`` that
    supplies predicted transitions and uses the oracle reward (the env's
    ``compute_reward``); it has no standalone ``default_params``.
    """

    def __init__(
        self,
        act_dim: int,
        reward_mode: Literal["area", "analytic"] = "analytic",
    ):
        self._act_dim = act_dim
        self._reward_mode = reward_mode

    @property
    def default_params(self) -> PlantEnvParams:
        raise RuntimeError(
            "PlantEnv has no default params — construct PlantEnvParams "
            "from the loaded dataset and pass it explicitly."
        )

    def observation_space(self, params: PlantEnvParams) -> spaces.Box:
        return spaces.Box(
            low=params.area_min,
            high=params.area_max,
            shape=(1,),
            dtype=jnp.float32,
        )

    def action_space(self, params: PlantEnvParams) -> spaces.Box:
        return spaces.Box(
            low=params.act_low,
            high=params.act_high,
            shape=(self._act_dim,),
            dtype=jnp.float32,
        )

    def reset_env(
        self, key: jax.Array, params: PlantEnvParams
    ) -> tuple[jax.Array, PlantEnvState]:
        """Sample a random episode-start log-area from the dataset."""
        n = params.init_areas.shape[0]
        idx = jax.random.randint(key, (), 0, n)
        area = params.init_areas[idx].reshape(1)
        state = PlantEnvState(obs=area, next_obs=jnp.zeros_like(area), time=0)
        return area, state

    def get_state(
        self,
        obs: jax.Array,
        last_action: jax.Array | None = None,
        time: int | None = None,
        next_obs: jax.Array | None = None,
    ) -> PlantEnvState:
        assert time is not None
        assert next_obs is not None
        return PlantEnvState(obs=obs, next_obs=next_obs, time=time)

    def step_env(
        self,
        key: jax.Array,
        state: PlantEnvState,
        action: jax.Array,
        params: PlantEnvParams,
    ) -> tuple[jax.Array, PlantEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        next_obs = state.next_obs
        reward = self.compute_reward(state.obs, action, next_obs)
        terminated = jnp.asarray(state.time + 1 >= params.max_steps_in_episode)
        new_state = PlantEnvState(
            obs=next_obs, next_obs=jnp.zeros_like(next_obs), time=state.time + 1
        )
        return next_obs, new_state, reward, terminated, {}

    def compute_reward(self, obs: jax.Array, action: jax.Array, next_obs: jax.Array) -> jax.Array:
        """Reward callable for external use (e.g. visualization)."""
        growth = (next_obs - obs)[..., 0]
        if self._reward_mode == "analytic":
            power = _P0 + _P_SLOPE * action[..., 0]
            return growth - 1 / _EP_LEN * (jnp.log(power * _HOURS) - jnp.log(_E_CONST_WH))
        return growth

    def get_obs(self, state: PlantEnvState, params=None, key=None) -> jax.Array:
        return state.obs

    def is_terminal(self, state: PlantEnvState, params: PlantEnvParams) -> jax.Array:
        return state.time >= params.max_steps_in_episode

    def state_space(self, params: PlantEnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "obs": self.observation_space(params),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def num_actions(self) -> int:
        return self._act_dim
