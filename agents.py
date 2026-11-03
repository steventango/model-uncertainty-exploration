import os
import time
from functools import partial
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

import plotting
import validation
from logger import log_validation
from model_env import reset_weights
from models import make_batched_model, make_batched_rngs, make_batched_train_model
from ppo import (
    make_batched_train,
    make_batched_train_state,
    make_train_state,
    make_update,
    select_config_train_state,
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

    def explore_policy(self, state):
        return state.explore

    def eval_policy(self, state):
        return state.eval

    def update(self, state, data, key):
        val_keys, plot_seed, unc_seed, ts_seed, train_seed = vsplit(key, 5)
        step = data.size

        t0 = time.perf_counter()
        history = self._train_model(
            state.models,
            state.model_train_state,
            data.buffer,
            jnp.asarray(data.size),
            state.rngs,
        )
        jax.block_until_ready(history)
        log_time(self.loggers, "time/model_train_s", t0, step)
        for b, logger in enumerate(self.loggers):
            logger.log_loss_history(seed_slice(history, b), data.round)

        if self.val_data is not None:
            t0 = time.perf_counter()
            log_validation(
                self.loggers,
                self._val_metrics,
                state.models,
                *self.val_data,
                step,
                val_keys,
            )
            log_time(self.loggers, "time/validation_s", t0, step)

        if self.debug is not None:
            # Plots seed 0 only.
            t0 = time.perf_counter()
            model = unstack_train_state(state.models, 0)
            buffer = seed_slice(data.buffer, 0)
            base_env, env_name, seed_dir = self.debug
            validation.evaluate_validation(
                model,
                base_env,
                self.env_params,
                env_name,
                *self.val_data,
                buffer,
                data.size,
                plot_seed[0],
                data.round,
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
                data.size,
                data.round,
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
            model=state.models, alpha=self.alphas, beta=self.betas
        )
        if self.reset_source != "env":
            weights = reset_weights(
                data.buffer.terminated,
                data.buffer.truncated,
                data.size,
                self.reset_source,
            )
            params = params.replace(init_obs=data.buffer.obs, init_weights=weights)
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
                    data.round,
                    self.ppo_config["NUM_ENVS"],
                    self.ppo_config["NUM_STEPS"],
                    int(self.ppo_config["TOTAL_TIMESTEPS"]),
                )
        trained = out["runner_state"][0]
        return state._replace(
            explore=select_config_train_state(trained, self.EXPLORE),
            eval=select_config_train_state(trained, self.EVAL),
        )

    def save(self, state, seed_dir, b):
        save_checkpoint(
            seed_dir, state.models, state.explore, state.eval, b, self.solver
        )
