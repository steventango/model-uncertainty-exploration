from __future__ import annotations

from functools import partial
from typing import Any, Generic, TypeVar

import jax
from gymnax.environments import environment

TEnvState = TypeVar("TEnvState", bound=environment.EnvState)
TEnvParams = TypeVar("TEnvParams", bound=environment.EnvParams)


class Environment(
    environment.Environment[TEnvState, TEnvParams],
    Generic[TEnvState, TEnvParams],
):
    @property
    def default_params(self) -> TEnvParams:
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            key_step, state, action, params
        )
        done = terminated | truncated
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on done
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        return obs, state, reward, terminated, truncated, info

    def step_env(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition."""
        raise NotImplementedError
