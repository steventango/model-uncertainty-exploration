import os
import time
from functools import partial
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from ppo import (
    make_train_state,
    make_update,
    unstack_train_state,
)


@partial(jax.jit, static_argnums=1)
def vsplit(keys, num=2):
    return jax.vmap(lambda k: jax.random.split(k, num), out_axes=1)(keys)


def seed_slice(tree, idx):
    """Index every leaf of a batched pytree, e.g. pick one seed (or (seed, config))."""
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def log_time(loggers, tag, t0, step):
    seconds = time.perf_counter() - t0
    for logger in loggers:
        logger.log_scalar(tag, seconds, step)


class Data(NamedTuple):
    """What an agent learns from; every array has a leading seed axis."""

    buffer: Any  # Transition, (B, N, ...); the first `size` rows are filled
    size: int
    latest: Any  # Transition, (B, T, E, ...) from this round's rollout; None offline
    round: int


class Agent(Protocol):
    """Learns from real-env data. All state is batched over a leading seed axis."""

    # The real-env rollout updates the explore policy's normalizers, and update()
    # learns from data.latest.
    on_policy: bool

    def init(self, key) -> Any: ...

    def explore_policy(self, state) -> Any:
        """PPO train state that acts in the real env."""

    def eval_policy(self, state) -> Any:
        """PPO train state that is evaluated in the real env."""

    def update(self, state, data: Data, key) -> Any: ...

    def save(self, state, seed_dir: str, b: int) -> None: ...


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


class ModelFreeState(NamedTuple):
    policy: Any


class ModelFreeAgent:
    """PPO on the real-env rollout of each round."""

    on_policy = True

    def __init__(self, config, env, env_params, loggers, solver):
        if config["NUM_STEPS"] % config["NUM_MINIBATCHES"]:
            raise ValueError(
                f"--model-free: --ppo.num-minibatches ({config['NUM_MINIBATCHES']})"
                f" must divide the real steps per rollout ({config['NUM_STEPS']})"
            )
        self.config = config
        self.env = env
        self.env_params = env_params
        self.loggers = loggers
        self.solver = solver
        ppo_update = make_update(
            {
                **config,
                "MINIBATCH_SIZE": config["NUM_STEPS"]
                * config["NUM_ENVS"]
                // config["NUM_MINIBATCHES"],
            }
        )

        def train(train_state, traj_batch, rng):
            _, (total, (value, actor, entropy)) = ppo_update(
                train_state, traj_batch, rng
            )
            losses = {
                "total": total,
                "value": value,
                "actor": actor,
                "entropy": entropy,
            }
            return jax.tree_util.tree_map(jnp.mean, losses)

        self._train = nnx.jit(nnx.vmap(train))

    def init(self, key):
        return ModelFreeState(
            make_batched_policy(self.config, self.env, self.env_params, key)
        )

    def explore_policy(self, state):
        return state.policy

    def eval_policy(self, state):
        return state.policy

    def update(self, state, data, key):
        t0 = time.perf_counter()
        losses = self._train(state.policy, data.latest, key)
        jax.block_until_ready(losses)
        log_time(self.loggers, f"time/{self.solver}_train_s", t0, data.size)
        for b, logger in enumerate(self.loggers):
            for name, loss in seed_slice(losses, b).items():
                logger.log_scalar(f"{self.solver}/{name}", float(loss), data.size)
        return state

    def save(self, state, seed_dir, b):
        save_checkpoint(seed_dir, None, state.policy, state.policy, b, self.solver)
