import argparse
import os
from datetime import datetime

from flax import nnx
import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
import optax

import evaluation
from data import collate_rollout
from env_config import SUPPORTED_ENVS
from logger import ExperimentLogger
from model import DynamicsModel, train_model
from model_env import ModelEnvironment
from networks import ENN
import plotting
from ppo import (
    make_batched_train,
    make_batched_train_state,
    make_rollout,
    make_train_state,
    unstack_train_state,
)
import validation
from wrappers import ClipAction, LogWrapper, VecEnv

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"


def _wrap_env(env, discrete):
    wrapped = LogWrapper(env)
    if not discrete:
        wrapped = ClipAction(wrapped)
    return VecEnv(wrapped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0, help="Exploit weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Explore weight")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        choices=SUPPORTED_ENVS,
        help="Gymnax environment name",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: runs/{env}_{timestamp})",
    )
    parser.add_argument(
        "--model_env_mode",
        type=str,
        default="mean",
        choices=["mean", "sample"],
        help="Model env transition mode: mean (base net) or sample (epinet at z[0])",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug plots (true vs predicted, uncertainty heatmaps)",
    )
    args = parser.parse_args()

    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "HIDDEN_DIM": 64,
        "ACTIVATION": "tanh",
        "ENV_NAME": args.env,
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "SEED": args.seed,
        "DEBUG": args.debug,
    }
    rollout_config = config.copy()
    rollout_config["NUM_ENVS"] = 1
    rollout_config["DATASET_SIZE"] = 10000
    eval_config = rollout_config.copy()
    eval_config["NUM_ENVS"] = 100

    rng = jax.random.key(config["SEED"])

    base_env, env_params = gymnax.make(rollout_config["ENV_NAME"])
    action_space = base_env.action_space(env_params)
    discrete = isinstance(action_space, spaces.Discrete)
    action_dim = action_space.n if discrete else action_space.shape[0]
    env = _wrap_env(base_env, discrete)
    rollout_config["TOTAL_TIMESTEPS"] = env_params.max_steps_in_episode * 10
    rollout_config["NUM_STEPS"] = env_params.max_steps_in_episode // 10
    eval_config["NUM_STEPS"] = env_params.max_steps_in_episode
    model_config = {
        "LR": 1e-3,
        "HIDDEN_DIM": 64,
        "LEARNABLE_HIDDEN_DIM": 15,
        "PRIOR_HIDDEN_DIM": 5,
        "INDEX_DIM": 8,
        "ACTIVATION": "tanh",
        "UPDATE_STEPS": 10000,
        "MINIBATCH_SIZE": rollout_config["NUM_STEPS"],
    }

    env_name = rollout_config["ENV_NAME"]
    (
        val_obs,
        val_act,
        val_true_delta_obs,
        val_true_reward,
        val_true_terminated,
    ) = validation.generate_validation_data(base_env, env_params, env_name)

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    train_state = make_train_state(config, env, env_params, rngs)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, rollout_config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)

    # MOCK ROLLOUT
    _, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, _rng)
    _rollout = nnx.jit(make_rollout(rollout_config, env, env_params, training=False))
    runner_state, traj_batch = _rollout(runner_state)
    sample = collate_rollout(traj_batch)

    # INIT DATASET
    dataset = jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (rollout_config["DATASET_SIZE"],) + x.shape[1:], dtype=x.dtype
        ),
        sample,
    )
    pointer = 0

    # INIT MODEL
    obs_dim = env.observation_space(env_params).shape[0]
    in_features = obs_dim + action_dim
    out_features = obs_dim + 2
    enn = ENN(
        in_features,
        model_config["HIDDEN_DIM"],
        model_config["LEARNABLE_HIDDEN_DIM"],
        model_config["PRIOR_HIDDEN_DIM"],
        out_features,
        model_config["INDEX_DIM"],
        rngs=rngs,
    )
    model = DynamicsModel(
        enn, in_features, obs_dim, act_dim=action_space.n if discrete else None
    )
    tx = optax.adamw(model_config["LR"], weight_decay=1e-4)
    not_prior_params = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))
    optimizer = nnx.Optimizer(model, tx, wrt=not_prior_params)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        delta_next_state_loss=nnx.metrics.Average("delta_next_state_loss"),
        reward_loss=nnx.metrics.Average("reward_loss"),
        terminated_loss=nnx.metrics.Average("terminated_loss"),
    )

    runner_state = (train_state, env_state, obsv, _rng)

    log_dir = args.log_dir or (
        f"runs/{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger = ExperimentLogger(log_dir)
    num_rollouts = int(
        rollout_config["TOTAL_TIMESTEPS"]
        // rollout_config["NUM_STEPS"]
        // rollout_config["NUM_ENVS"]
    )
    logger.log_hparams(
        args=vars(args),
        ppo=config,
        rollout=rollout_config,
        eval=eval_config,
        model=model_config,
        run={
            "num_rollouts": num_rollouts,
            "discrete": discrete,
            "action_dim": action_dim,
        },
    )

    _eval_rollout = nnx.jit(make_rollout(eval_config, env, env_params, training=False))

    model_env = ModelEnvironment(
        env,
        env_params,
        prediction_mode=args.model_env_mode,
    )
    model_env = _wrap_env(model_env, discrete)
    model_env_params = model_env.default_params.with_model(model)
    EXPLORE, EVAL = 0, 1
    alphas = jnp.array([args.alpha, 1.0])
    betas = jnp.array([args.beta, 0.0])
    num_configs = alphas.shape[0]
    batched_train_jit = nnx.jit(
        make_batched_train(model_env, model_env_params, config)
    )
    for j in range(num_rollouts):
        # ROLLOUT
        runner_state, traj_batch = _rollout(runner_state)

        # UPDATE DATASET
        traj_batch = collate_rollout(traj_batch)
        dataset = jax.tree_util.tree_map(
            lambda old, new: jax.lax.dynamic_update_slice_in_dim(
                old, new, pointer, axis=0
            ),
            dataset,
            traj_batch,
        )
        logger.log_dataset(traj_batch, pointer)
        pointer += traj_batch.obs.shape[0]

        # TRAIN MODEL
        history = train_model(
            model,
            optimizer,
            metrics,
            dataset,
            model_config["UPDATE_STEPS"],
            pointer,
            model_config["MINIBATCH_SIZE"],
            rngs=rngs,
        )

        logger.log_loss_history(history, j)

        if pointer > 0:
            rng, _rng = jax.random.split(rng)
            dyn_mae, rew_mae, term_bce, term_f1, _ = validation.evaluate_validation(
                model,
                base_env,
                env_params,
                env_name,
                val_obs,
                val_act,
                val_true_delta_obs,
                val_true_reward,
                val_true_terminated,
                dataset,
                pointer,
                _rng,
                j,
                log_dir,
                plot=config["DEBUG"],
            )
            logger.log_validation_metrics(dyn_mae, rew_mae, term_bce, term_f1, j)

        if config["DEBUG"]:
            rng, _rng = jax.random.split(rng)
            plotting.evaluate_and_plot_uncertainty(
                model,
                base_env,
                env_params,
                env_name,
                _rng,
                dataset,
                pointer,
                j,
                log_dir,
            )

        rng, _rng = jax.random.split(rng)
        batched_train_state = make_batched_train_state(
            config, model_env, model_env_params, _rng, num_configs
        )
        rng, _rng = jax.random.split(rng)
        out = batched_train_jit(
            batched_train_state,
            alphas,
            betas,
            jax.random.split(_rng, num_configs),
        )
        batched_out_train_state = out["runner_state"][0]
        explore_train_state = unstack_train_state(batched_out_train_state, EXPLORE)
        eval_train_state = unstack_train_state(batched_out_train_state, EVAL)

        runner_state = (
            explore_train_state,
            runner_state[1],
            runner_state[2],
            runner_state[3],
        )

        for tag, index in (("ppo/explore_return", EXPLORE), ("ppo/eval_return", EVAL)):
            logger.log_ppo_returns(
                jax.tree_util.tree_map(lambda x: x[index], out["metrics"]),
                tag,
                j,
                config["NUM_ENVS"],
                config["NUM_STEPS"],
                int(config["TOTAL_TIMESTEPS"]),
            )

        rng, _rng = jax.random.split(rng)
        mean_return = evaluation.evaluate_policy(
            eval_config,
            env,
            env_params,
            eval_train_state,
            _rng,
            rollout_fn=_eval_rollout,
        )
        logger.log_eval_return(pointer, mean_return)

    logger.close()


if __name__ == "__main__":
    main()
