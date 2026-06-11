from dataclasses import dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from model import DynamicsModel


@struct.dataclass
class ModelEnvState(environment.EnvState):
    obs: jnp.ndarray
    terminated: jnp.ndarray
    time: int
    z: jnp.ndarray


@dataclass
class ModelEnvParams:
    env_params: environment.EnvParams
    max_steps_in_episode: int
    model: DynamicsModel | None = None

    def with_model(self, model: DynamicsModel) -> "ModelEnvParams":
        return replace(self, model=model)


class ModelEnvironment(environment.Environment[ModelEnvState, ModelEnvParams]):
    def __init__(
        self,
        env: environment.Environment,
        env_params: environment.EnvParams,
        samples: int = 10,
        alpha: float = 1.0,
        beta: float = 0.1,
        prediction_mode: Literal["mean", "sample"] = "mean",
    ):
        if prediction_mode not in ("mean", "sample"):
            raise ValueError(
                f"prediction_mode must be 'mean' or 'sample', got {prediction_mode!r}"
            )
        self._real_env = env
        self._real_env_params = env_params
        self.alpha = alpha
        self.beta = beta
        self.samples = samples
        self.prediction_mode = prediction_mode

    @property
    def default_params(self) -> ModelEnvParams:
        return ModelEnvParams(
            env_params=self._real_env_params,
            max_steps_in_episode=self._real_env_params.max_steps_in_episode,
        )

    def step(
        self,
        key: jax.Array,
        state: ModelEnvState,
        action: int | float | jax.Array,
        params: ModelEnvParams | None = None,
    ) -> tuple[
        jax.Array, ModelEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]
    ]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, info = self.step_env(
            key_step, state, action, params
        )
        truncated = state_st.time >= params.max_steps_in_episode
        done = terminated | truncated
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        info = {**info, "next_obs": obs_st}

        return obs, state, reward, terminated, truncated, info

    def reset(
        self, key: jax.Array, params: ModelEnvParams | None = None
    ) -> tuple[jax.Array, ModelEnvState]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params

        # Reset
        obs, state = self.reset_env(key, params)

        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: ModelEnvState,
        action: int | float | jax.Array,
        params: ModelEnvParams,
    ) -> tuple[jax.Array, ModelEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        del key
        model = params.model
        x = model.single_input(state.obs, action)
        y_base, y_samples = jax.vmap(model.__call__, in_axes=(None, 0))(x, state.z)
        if self.prediction_mode == "mean":
            y = y_base[0]
        else:
            y = y_samples[0]
        r_intrinsic = y_samples.std(axis=0).mean()

        delta_obs = model.denormalize_delta_obs(y[..., :-2])
        obs = state.obs + delta_obs
        obs = jnp.clip(
            obs,
            self._real_env.observation_space(params.env_params).low,
            self._real_env.observation_space(params.env_params).high,
        )
        r_exploit = model.denormalize_reward(y[..., -2])
        r = self.alpha * r_exploit + self.beta * r_intrinsic
        terminated = jax.nn.sigmoid(y[..., -1]) > 0.5
        state = ModelEnvState(
            obs=obs, terminated=terminated, time=state.time + 1, z=state.z
        )
        return obs, state, r, terminated, {}

    def reset_env(
        self, key: jax.Array, params: ModelEnvParams
    ) -> tuple[jax.Array, ModelEnvState]:
        model = params.model
        key, key_reset, key_z = jax.random.split(key, 3)
        obs, _ = self._real_env.reset_env(key_reset, params.env_params)
        z = jax.random.normal(key_z, (self.samples, model.index_dim))
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
