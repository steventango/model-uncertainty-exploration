"""MountainCar-v0 with a noisy-TV action.

Action 3 has the physics of action 1 (no acceleration) and also writes
N(0, noise_std^2) into an extra observation dimension; every other action
writes 0 there. Obs = [position, velocity, noise]. The noise is independent of
the state and action history, so the next noise value is pure aleatoric noise
and the Markov state stays fully observed.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.classic_control import mountain_car


@struct.dataclass
class EnvState(mountain_car.EnvState):
    noise: jax.Array


@struct.dataclass
class EnvParams(mountain_car.EnvParams):
    noise_std: float = 1.0


NOISY_ACTION = 3


class NoisyMountainCar(mountain_car.MountainCar):
    def __init__(self):
        super().__init__()
        self._base = mountain_car.MountainCar()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        noisy = action == NOISY_ACTION
        base_state = mountain_car.EnvState(
            position=state.position, velocity=state.velocity, time=state.time
        )
        _, base_state, reward, done, info = self._base.step_env(
            key, base_state, jnp.where(noisy, 1, action), params
        )
        noise = jnp.where(noisy, params.noise_std * jax.random.normal(key), 0.0)
        state = EnvState(
            position=base_state.position,
            velocity=base_state.velocity,
            time=base_state.time,
            noise=noise,
        )
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        _, base_state = self._base.reset_env(key, params)
        state = EnvState(
            position=base_state.position,
            velocity=base_state.velocity,
            time=base_state.time,
            noise=jnp.array(0.0),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        return jnp.array([state.position, state.velocity, state.noise])

    @property
    def name(self) -> str:
        return "NoisyMountainCar-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        low = jnp.array([params.min_position, -params.max_speed, -jnp.inf])
        high = jnp.array([params.max_position, params.max_speed, jnp.inf])
        return spaces.Box(low, high, (3,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        base = self._base.state_space(params).spaces
        return spaces.Dict(
            {**base, "noise": spaces.Box(-jnp.inf, jnp.inf, (), dtype=jnp.float32)}
        )
