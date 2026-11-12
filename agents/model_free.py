import time
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from gymnax.environments import environment

from agents.base import (
    Config,
    Env,
    Experience,
    PolicyState,
    make_batched_policy,
    save_checkpoint,
)
from batching import seed_slice
from logger import ExperimentLogger, log_time
from ppo import (
    make_update,
)


class ModelFreeState(NamedTuple):
    policy: PolicyState


class ModelFreeAgent:
    """PPO on each real-env rollout."""

    on_policy: ClassVar[bool] = True

    def __init__(
        self,
        config: Config,
        env: Env,
        env_params: environment.EnvParams,
        loggers: list[ExperimentLogger],
        solver: str,
    ):
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
            _, (total, aux) = ppo_update(train_state, traj_batch, rng)
            losses = {**jax.tree_util.tree_map(jnp.mean, aux), "total": total.mean()}
            losses["approx_kl_max"] = aux["approx_kl"].max()
            return losses

        self._train = nnx.jit(nnx.vmap(train))

    def init(self, key: jax.Array) -> ModelFreeState:
        return ModelFreeState(
            make_batched_policy(self.config, self.env, self.env_params, key)
        )

    def explore_policy(self, agent_state: ModelFreeState) -> PolicyState:
        return agent_state.policy

    def eval_policy(self, agent_state: ModelFreeState) -> PolicyState:
        return agent_state.policy

    def update(
        self, agent_state: ModelFreeState, experience: Experience, key: jax.Array
    ) -> ModelFreeState:
        t0 = time.perf_counter()
        losses = self._train(agent_state.policy, experience.latest, key)
        jax.block_until_ready(losses)
        log_time(self.loggers, f"time/{self.solver}_train_s", t0, experience.size)
        for b, logger in enumerate(self.loggers):
            for name, loss in seed_slice(losses, b).items():
                logger.log_scalar(f"{self.solver}/{name}", float(loss), experience.size)
        return agent_state

    def save(self, agent_state: ModelFreeState, seed_dir: str, b: int) -> None:
        save_checkpoint(
            seed_dir, None, agent_state.policy, agent_state.policy, b, self.solver
        )
