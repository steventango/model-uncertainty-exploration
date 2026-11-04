import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

import plotting
import validation
from agents.base import make_batched_policy, save_checkpoint
from batching import seed_slice, vsplit
from logger import log_time, log_validation
from model_env import reset_weights
from models import make_batched_model, make_batched_rngs, make_batched_train_model
from ppo import (
    make_batched_train,
    make_batched_train_state,
    select_config_train_state,
    unstack_train_state,
)


class ModelBasedState(NamedTuple):
    models: Any
    model_train_state: Any
    rngs: Any
    explore: Any
    eval: Any


class ModelBasedAgent:
    """Fit a world model to the data, then train PPO inside it.

    PPO trains two configs per seed: explore (alpha, beta from the args) acts in
    the real env, and eval (alpha=1, beta=0, no bonus) is evaluated.
    """

    on_policy = False
    EXPLORE, EVAL = 0, 1

    def __init__(
        self,
        *,
        model_name,
        model_config,
        model_dims,
        predict_reward_terminated,
        ppo_config,
        env,
        env_params,
        model_env,
        reset_source,
        alpha,
        beta,
        loggers,
        solver,
        val_data=None,
        debug=None,
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
        self.val_data = val_data
        self.debug = debug
        self._train_model = make_batched_train_model(
            model_name, model_config["UPDATE_STEPS"], model_config["MINIBATCH_SIZE"]
        )
        self._val_metrics = validation.make_batched_validation_metrics()
        self._train_ppo = nnx.jit(
            make_batched_train(model_env, model_env.default_params, ppo_config)
        )

    def init(self, key):
        policy_key, model_key, rngs_key, val_key = vsplit(key, 4)
        policy = make_batched_policy(
            self.ppo_config, self.env, self.env_params, policy_key
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
        return ModelBasedState(
            models, model_train_state, make_batched_rngs(rngs_key), policy, policy
        )

    def explore_policy(self, agent_state):
        return agent_state.explore

    def eval_policy(self, agent_state):
        return agent_state.eval

    def update(self, agent_state, experience, key):
        val_keys, plot_seed, unc_seed, ts_seed, train_seed = vsplit(key, 5)
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

        num_configs = self.alphas.shape[0]
        train_states = make_batched_train_state(
            self.ppo_config,
            self.model_env,
            self.model_env.default_params,
            vsplit(ts_seed, num_configs).T,  # (B, C)
        )
        # model varies per seed (B axis); alpha/beta vary per config (C axis).
        params = self.model_env.default_params.replace(
            model=agent_state.models, alpha=self.alphas, beta=self.betas
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
        trained = out["runner_state"][0]
        return agent_state._replace(
            explore=select_config_train_state(trained, self.EXPLORE),
            eval=select_config_train_state(trained, self.EVAL),
        )

    def save(self, agent_state, seed_dir, b):
        save_checkpoint(
            seed_dir,
            agent_state.models,
            agent_state.explore,
            agent_state.eval,
            b,
            self.solver,
        )
