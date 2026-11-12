import time
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from gymnax.environments import environment

import plotting
import ppo
import sac
import validation
from agents.base import (
    Config,
    Env,
    Experience,
    PolicyState,
    make_batched_policy,
    save_checkpoint,
)
from batching import seed_slice, vsplit
from config import RNDConfig
from logger import ExperimentLogger, log_time, log_validation
from model_env import ModelEnvironment, ResetSource, reset_weights
from models import make_batched_model, make_batched_rngs, make_batched_train_model
from models.base import WorldModel
from ppo import (
    select_config_train_state,
    unstack_train_state,
)
from rnd import RND, make_rnd, make_train_rnd

# obs, act, true delta obs, true reward, true terminated
ValData = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


class ModelBasedState(NamedTuple):
    models: WorldModel
    model_train_state: (
        Any  # model-specific: (Optimizer, MultiMetric) for ENN, None for BLR
    )
    rngs: nnx.Rngs
    explore: PolicyState
    eval: PolicyState
    rnd: RND | None = None  # explore_bonus="rnd"
    rnd_optimizer: nnx.Optimizer | None = None


class ModelBasedAgent:
    """Fit a world model to the data, then train PPO inside it.

    PPO trains two configs per seed: explore (alpha, beta from the args) acts in
    the real env, and eval (alpha=1, beta=0, no bonus) is evaluated.
    """

    on_policy: ClassVar[bool] = False
    EXPLORE: ClassVar[int] = 0
    EVAL: ClassVar[int] = 1

    def __init__(
        self,
        *,
        model_name: str,
        model_config: Config,
        model_dims: tuple[int, int, int, int],  # in_features, obs, out_features, act
        predict_reward_terminated: bool,
        ppo_config: Config,
        env: Env,
        env_params: environment.EnvParams,
        model_env: ModelEnvironment,
        reset_source: ResetSource,
        alpha: float,
        beta: float,
        loggers: list[ExperimentLogger],
        solver: str,
        rnd_config: RNDConfig | None = None,
        val_data: ValData | None = None,
        debug: tuple[Env, str, str] | None = None,  # base_env, env name, seed-0 dir
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.model_dims = model_dims
        self.predict_reward_terminated = predict_reward_terminated
        self.ppo_config = ppo_config
        self.env = env
        self.env_params = env_params
        self.model_env = model_env
        self.reset_source = reset_source
        self.alphas = jnp.array([alpha, 1.0])
        self.betas = jnp.array([beta, 0.0])
        self.loggers = loggers
        self.solver = solver
        self.rnd_config = rnd_config
        self.val_data = val_data
        self.debug = debug
        self._train_model = make_batched_train_model(
            model_name, model_config["UPDATE_STEPS"], model_config["MINIBATCH_SIZE"]
        )
        self._val_metrics = validation.make_batched_validation_metrics()
        if rnd_config is not None:
            self._train_rnd = nnx.jit(
                nnx.vmap(make_train_rnd(rnd_config), in_axes=(0, 0, 0, None, 0))
            )
        self.solver_module = {"ppo": ppo, "sac": sac}[solver]
        self._train_ppo = nnx.jit(
            self.solver_module.make_batched_train(
                model_env, model_env.default_params, ppo_config
            )
        )

    def init(self, key: jax.Array) -> ModelBasedState:
        policy_key, model_key, rngs_key, val_key, rnd_key = vsplit(key, 5)
        policy = make_batched_policy(
            self.solver_module.make_train_state,
            self.ppo_config,
            self.env,
            self.env_params,
            policy_key,
        )
        models, model_train_state = make_batched_model(
            self.model_name,
            self.model_config,
            *self.model_dims,
            model_key,
            predict_reward_terminated=self.predict_reward_terminated,
        )
        if self.val_data is not None:
            log_validation(
                self.loggers, self._val_metrics, models, *self.val_data, 0, val_key
            )
        rnd = rnd_optimizer = None
        if self.rnd_config is not None:
            obs_dim = self.model_dims[1]

            @nnx.vmap
            def build_rnd(key):
                return make_rnd(self.rnd_config, obs_dim, nnx.Rngs(params=key))

            rnd, rnd_optimizer = build_rnd(rnd_key)
        return ModelBasedState(
            models,
            model_train_state,
            make_batched_rngs(rngs_key),
            policy,
            policy,
            rnd,
            rnd_optimizer,
        )

    def explore_policy(self, agent_state: ModelBasedState) -> PolicyState:
        return agent_state.explore

    def eval_policy(self, agent_state: ModelBasedState) -> PolicyState:
        return agent_state.eval

    def update(
        self, agent_state: ModelBasedState, experience: Experience, key: jax.Array
    ) -> ModelBasedState:
        val_keys, plot_seed, unc_seed, ts_seed, train_seed, rnd_key = vsplit(key, 6)
        step = experience.size

        t0 = time.perf_counter()
        history = self._train_model(
            agent_state.models,
            agent_state.model_train_state,
            experience.buffer,
            jnp.asarray(experience.size),
            agent_state.rngs,
        )
        jax.block_until_ready(history)
        log_time(self.loggers, "time/model_train_s", t0, step)
        for b, logger in enumerate(self.loggers):
            logger.log_loss_history(seed_slice(history, b), experience.rollout)

        if self.val_data is not None:
            t0 = time.perf_counter()
            log_validation(
                self.loggers,
                self._val_metrics,
                agent_state.models,
                *self.val_data,
                step,
                val_keys,
            )
            log_time(self.loggers, "time/validation_s", t0, step)

        if self.debug is not None:
            # Plots seed 0 only.
            t0 = time.perf_counter()
            model = unstack_train_state(agent_state.models, 0)
            buffer = seed_slice(experience.buffer, 0)
            base_env, env_name, seed_dir = self.debug
            validation.evaluate_validation(
                model,
                base_env,
                self.env_params,
                env_name,
                *self.val_data,
                buffer,
                experience.size,
                plot_seed[0],
                experience.rollout,
                seed_dir,
                plot=True,
            )
            plotting.evaluate_and_plot_uncertainty(
                model,
                base_env,
                self.env_params,
                env_name,
                unc_seed[0],
                buffer,
                experience.size,
                experience.rollout,
                seed_dir,
            )
            log_time(self.loggers, "time/validation_eval_plot_s", t0, step)

        if self.rnd_config is not None:
            rnd_losses = self._train_rnd(
                agent_state.rnd,
                agent_state.rnd_optimizer,
                experience.buffer.info["next_obs"],
                jnp.asarray(experience.size),
                rnd_key,
            )
            for b, logger in enumerate(self.loggers):
                logger.log_scalar("rnd/loss", float(rnd_losses[b, -1]), step)

        num_configs = self.alphas.shape[0]
        train_states = self.solver_module.make_batched_train_state(
            self.ppo_config,
            self.model_env,
            self.model_env.default_params,
            vsplit(ts_seed, num_configs).T,  # (B, C)
        )
        # model varies per seed (B axis); alpha/beta vary per config (C axis).
        params = self.model_env.default_params.replace(
            env_params=self.model_env.seed_env_params
            if self.model_env.seed_env_params is not None
            else self.model_env.default_params.env_params,
            model=agent_state.models,
            rnd=agent_state.rnd,
            alpha=self.alphas,
            beta=self.betas,
        )
        if self.reset_source != "env":
            weights = reset_weights(
                experience.buffer.terminated,
                experience.buffer.truncated,
                experience.size,
                self.reset_source,
            )
            params = params.replace(
                init_obs=experience.buffer.obs, init_weights=weights
            )
        if self.solver == "sac" and self.ppo_config["REAL_RATIO"] > 0:
            buffer = experience.buffer
            real = (
                buffer.obs,
                buffer.action,
                buffer.info["next_obs"],
                buffer.terminated,
            )
            count = jnp.full(buffer.obs.shape[0], experience.size)
            params = params.replace(real=real, real_count=count)
        t0 = time.perf_counter()
        out = self._train_ppo(train_states, params, vsplit(train_seed, num_configs).T)
        jax.block_until_ready(out)
        log_time(self.loggers, f"time/{self.solver}_train_s", t0, step)

        for b, logger in enumerate(self.loggers):
            for tag, index in (
                (f"{self.solver}/explore_return", self.EXPLORE),
                (f"{self.solver}/eval_return", self.EVAL),
            ):
                logger.log_ppo_returns(
                    seed_slice(out["metrics"], (b, index)),
                    tag,
                    experience.rollout,
                    self.ppo_config["NUM_ENVS"],
                    self.ppo_config["NUM_STEPS"],
                    int(self.ppo_config["TOTAL_TIMESTEPS"]),
                )
        for name, values in out.get("losses", {}).items():
            for b, logger in enumerate(self.loggers):
                for tag, index in (("explore", self.EXPLORE), ("eval", self.EVAL)):
                    logger.log_scalar(
                        f"{self.solver}/{tag}_{name}", float(values[b, index, -1]), step
                    )
        trained = out["runner_state"][0]
        return agent_state._replace(
            explore=select_config_train_state(trained, self.EXPLORE),
            eval=select_config_train_state(trained, self.EVAL),
        )

    def save(self, agent_state: ModelBasedState, seed_dir: str, b: int) -> None:
        save_checkpoint(
            seed_dir,
            agent_state.models,
            agent_state.explore,
            agent_state.eval,
            b,
            self.solver,
        )
