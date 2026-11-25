import os
from collections.abc import Callable
from typing import Any, ClassVar, NamedTuple, Protocol

import jax
import orbax.checkpoint as ocp
from flax import nnx
from gymnax.environments import environment

from batching import seed_slice
from models.base import WorldModel
from networks import ActorCritic
from normalization import NormalizeVecObs, NormalizeVecReward
from ppo import (
    Transition,
    make_train_state,
    unstack_train_state,
)
from wrappers import BraxGymnaxWrapper, GymnaxWrapper

Env = environment.Environment | GymnaxWrapper | BraxGymnaxWrapper
Config = dict[str, Any]
PolicyState = tuple[ActorCritic, nnx.Optimizer, NormalizeVecObs, NormalizeVecReward]
# make_rollout(config, env, env_params, training) -> rollout, where
# rollout(runner_state) -> (runner_state, traj_batch); see ppo.make_rollout.
RolloutFactory = Callable[[Config, Any, Any, bool], Callable]


class Experience(NamedTuple):
    """What an agent learns from; every array has a leading seed axis."""

    buffer: Transition  # (B, N, ...); the first `size` rows are filled
    size: int
    latest: Transition | None  # (B, T, E, ...) from the latest rollout; None offline
    rollout: int


class Agent[S, P](Protocol):
    """Learns from real-env data. All state is batched over a leading seed axis.

    S is the agent's state; P is the policy object that acts in the real env and rides
    in runner_state[0] (a PPO PolicyState for the model-based/model-free agents, an
    OpaxPolicy for OPAX). Its leaves keep the leading seed axis.

    Optional hook (not a Protocol member, so agents without it still conform):
    ``make_rollout(config, env, env_params, training)``, a RolloutFactory with the same
    contract as ppo.make_rollout: runner_state = (policy, env_state, obs, rng) for one
    seed, returning (runner_state, ppo.Transition with (T, E, ...) leaves). run_online
    and evaluation use it when the agent defines it, and ppo.make_rollout otherwise.
    """

    # The real-env rollout updates the explore policy's normalizers, and update()
    # learns from experience.latest.
    on_policy: ClassVar[bool]

    def init(self, key: jax.Array) -> S: ...

    def explore_policy(self, agent_state: S) -> P:
        """Policy that acts in the real env."""
        ...

    def eval_policy(self, agent_state: S) -> P:
        """Policy that is evaluated in the real env."""
        ...

    def update(self, agent_state: S, experience: Experience, key: jax.Array) -> S: ...

    def save(self, agent_state: S, seed_dir: str, b: int) -> None: ...


def make_batched_policy(
    config: Config, env: Env, env_params: environment.EnvParams, keys: jax.Array
) -> PolicyState:
    env_params = seed_slice(env_params, 0)

    @nnx.vmap
    def build(key):
        return make_train_state(config, env, env_params, nnx.Rngs(params=key))

    return build(keys)


def save_checkpoint(
    seed_dir: str,
    models: WorldModel | None,
    explore_train_state: PolicyState,
    eval_train_state: PolicyState,
    seed_idx: int,
    solver: str = "ppo",
) -> None:
    checkpointer = ocp.StandardCheckpointer()
    abs_seed_dir = os.path.abspath(seed_dir)

    if models is not None:
        _, model_state = nnx.split(unstack_train_state(models, seed_idx))
        checkpointer.save(
            os.path.join(abs_seed_dir, "checkpoint", "model"), model_state
        )

    for tag, batched_ts in [
        (f"{solver}_explore", explore_train_state),
        (f"{solver}_eval", eval_train_state),
    ]:
        network, _, normalize_vec_obs, _ = unstack_train_state(batched_ts, seed_idx)
        _, network_state = nnx.split(network)
        _, obs_norm_state = nnx.split(normalize_vec_obs)
        base = os.path.join(abs_seed_dir, "checkpoint", tag)
        checkpointer.save(os.path.join(base, "network"), network_state)
        checkpointer.save(os.path.join(base, "obs_norm"), obs_norm_state)

    checkpointer.wait_until_finished()
