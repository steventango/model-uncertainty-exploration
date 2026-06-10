from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from networks import ENN


@struct.dataclass
class ModelEnvState(environment.EnvState):
    obs: jnp.ndarray
    terminated: jnp.ndarray
    time: int
    z: jnp.ndarray


@struct.dataclass
class ModelEnvParams(environment.EnvParams):
    env_params: environment.EnvParams = environment.EnvParams()
    max_steps_in_episode: int = 1


class ModelEnvironment(environment.Environment[ModelEnvState, ModelEnvParams]):
    def __init__(
        self,
        env: environment.Environment,
        env_params: environment.EnvParams,
        model: ENN,
        samples: int = 10,
        alpha: float = 1.0,
        beta: float = 0.1,
    ):
        self._real_env = env
        self._real_env_params = env_params
        self._model = model
        self.alpha = alpha
        self.beta = beta
        self.samples = samples

    @property
    def default_params(self) -> ModelEnvParams:
        return ModelEnvParams(
            env_params=self._real_env_params,
            max_steps_in_episode=self._real_env_params.max_steps_in_episode,
        )

    def step_env(
        self,
        key: jax.Array,
        state: ModelEnvState,
        action: int | float | jax.Array,
        params: ModelEnvParams,
    ) -> tuple[jax.Array, ModelEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition."""
        x = jnp.concatenate([state.obs, jnp.atleast_1d(action)], axis=-1)
        y_base, y_samples = jax.vmap(
            self._model.__call__, in_axes=(None, 0)
        )(x, state.z)
        y = y_base[0]
        r_intrinsic = y_samples.std(axis=0).mean()

        obs = state.obs + y[..., :-2]
        obs = jnp.clip(
            obs,
            self._real_env.observation_space(params.env_params).low,
            self._real_env.observation_space(params.env_params).high,
        )
        r = self.alpha * y[..., -2] + self.beta * r_intrinsic
        terminated = jax.nn.sigmoid(y[..., -1]) > 0.5
        state = ModelEnvState(
            obs=obs, terminated=terminated, time=state.time + 1, z=state.z
        )
        return obs, state, r, terminated, {}

    def reset_env(
        self, key: jax.Array, params: ModelEnvParams
    ) -> tuple[jax.Array, ModelEnvState]:
        """Environment-specific reset."""
        key, key_obs, key_z = jax.random.split(key, 3)
        obs = self._real_env.observation_space(params.env_params).sample(key_obs)
        z = jax.random.normal(key_z, (self.samples, self._model.index_dim))
        state = ModelEnvState(
            obs=obs,
            terminated=jnp.bool_(False),
            time=0,
            z=z,
        )
        return obs, state

    def get_obs(self, state: ModelEnvState, params=None, key=None) -> jax.Array:
        """Applies observation function to state."""
        return state.obs

    def is_terminal(self, state: ModelEnvState, params: ModelEnvParams) -> jax.Array:
        """Check whether state transition is terminal."""
        return state.terminated

    def discount(self, state: ModelEnvState, params: ModelEnvParams) -> jax.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._real_env.num_actions

    def action_space(self, params: ModelEnvParams):
        """Action space of the environment."""
        return self._real_env.action_space(params.env_params)

    def observation_space(self, params: ModelEnvParams):
        """Observation space of the environment."""
        return self._real_env.observation_space(params.env_params)

    def state_space(self, params: ModelEnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
                "obs": self._real_env.observation_space(params.env_params),
                "terminated": spaces.Box(
                    low=False, high=True, shape=(), dtype=jnp.bool
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
