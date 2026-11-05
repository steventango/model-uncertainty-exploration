"""Drifter-Distractor (Linke et al., 2020, "Adapting Behavior via Intrinsic Reward").

Each of the four actions samples one target: y ~ N(mu_target, sigma^2_type), and
after every step each target mean takes a random-walk step N(0, xi^2_type),
clipped to [-mu_bound, mu_bound]. Target types (paper Table 1):

    constant:   mu0 ~ U(-50, 50), sigma^2 = 0, xi^2 = 0
    distractor: mu0 = 0,          sigma^2 = 1, xi^2 = 0
    drifter:    mu0 = 0,          sigma^2 = 0, xi^2 = 0.1

Obs = [y], the sample just drawn (0 at reset); the means are hidden state.
Reward is 0 and the env never terminates. The default episode is long enough
that the real process never resets within a run, so drift is continuing.

Switched variant: from step ``switch_step`` on, action a samples target
``switch_permutation[a]``. The paper's exact Table 2 permutation was not
available; the default (2, 3, 0, 1) changes the target type of every action.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

CONSTANT, DISTRACTOR, DRIFTER = 0, 1, 2
NUM_ACTIONS = 4


@struct.dataclass
class EnvState(environment.EnvState):
    mu: jax.Array  # (NUM_ACTIONS,) per-target mean


@struct.dataclass
class EnvParams(environment.EnvParams):
    target_types: tuple[int, ...] = struct.field(
        pytree_node=False, default=(DISTRACTOR, DISTRACTOR, DRIFTER, CONSTANT)
    )
    constant_range: float = 50.0
    distractor_var: float = 1.0
    drifter_step_var: float = 0.1
    mu_bound: float = 50.0
    switch_step: int = -1  # < 0 disables the switch
    switch_permutation: tuple[int, ...] = struct.field(
        pytree_node=False, default=(2, 3, 0, 1)
    )
    max_steps_in_episode: int = 10**6


def action_targets(time, params: EnvParams) -> jax.Array:
    """Target index sampled by each action at ``time``, shape (NUM_ACTIONS,)."""
    switched = (params.switch_step >= 0) & (time >= params.switch_step)
    return jnp.where(
        switched,
        jnp.asarray(params.switch_permutation),
        jnp.arange(NUM_ACTIONS),
    )


class DrifterDistractor(environment.Environment[EnvState, EnvParams]):
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def _types(self, params: EnvParams) -> jax.Array:
        return jnp.asarray(params.target_types)

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        key_sample, key_drift = jax.random.split(key)
        types = self._types(params)
        sample_var = jnp.where(types == DISTRACTOR, params.distractor_var, 0.0)
        step_var = jnp.where(types == DRIFTER, params.drifter_step_var, 0.0)

        targets = action_targets(state.time, params)
        target = targets[action]
        y = state.mu[target] + jnp.sqrt(sample_var[target]) * jax.random.normal(
            key_sample
        )
        mu = state.mu + jnp.sqrt(step_var) * jax.random.normal(
            key_drift, (NUM_ACTIONS,)
        )
        mu = jnp.clip(mu, -params.mu_bound, params.mu_bound)

        info = {"mu": state.mu[targets]}
        next_state = EnvState(time=state.time + 1, mu=mu)
        obs = jnp.reshape(y, (1,))
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(next_state),
            jnp.zeros(()),
            self.is_terminal(next_state, params),
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        types = self._types(params)
        mu0 = jax.random.uniform(
            key,
            (NUM_ACTIONS,),
            minval=-params.constant_range,
            maxval=params.constant_range,
        )
        mu = jnp.where(types == CONSTANT, mu0, 0.0)
        state = EnvState(time=0, mu=mu)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        return jnp.zeros((1,))

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.array(False)

    @property
    def name(self) -> str:
        return "DrifterDistractor-v0"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "mu": spaces.Box(
                    -params.mu_bound, params.mu_bound, (NUM_ACTIONS,), jnp.float32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
