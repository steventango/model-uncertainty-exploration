import os
from typing import Any, NamedTuple, Protocol

import orbax.checkpoint as ocp
from flax import nnx

from ppo import (
    make_train_state,
    unstack_train_state,
)


class Experience(NamedTuple):
    """What an agent learns from; every array has a leading seed axis."""

    buffer: Any  # Transition, (B, N, ...); the first `size` rows are filled
    size: int
    latest: Any  # Transition, (B, T, E, ...) from the latest rollout; None offline
    rollout: int


class Agent(Protocol):
    """Learns from real-env data. All state is batched over a leading seed axis."""

    # The real-env rollout updates the explore policy's normalizers, and update()
    # learns from experience.latest.
    on_policy: bool

    def init(self, key) -> Any: ...

    def explore_policy(self, agent_state) -> Any:
        """PPO train state that acts in the real env."""

    def eval_policy(self, agent_state) -> Any:
        """PPO train state that is evaluated in the real env."""

    def update(self, agent_state, experience: Experience, key) -> Any: ...

    def save(self, agent_state, seed_dir: str, b: int) -> None: ...


def make_batched_policy(config, env, env_params, keys):
    @nnx.vmap
    def build(key):
        return make_train_state(config, env, env_params, nnx.Rngs(params=key))

    return build(keys)


def save_checkpoint(
    seed_dir, models, explore_train_state, eval_train_state, seed_idx, solver="ppo"
):
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
