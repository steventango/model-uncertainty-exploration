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
from ppo import make_rollout, make_train, make_train_state
import validation
from wrappers import ClipAction, LogWrapper, VecEnv

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"


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
        "DEBUG": False,
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
    env = LogWrapper(base_env)
    if not discrete:
        env = ClipAction(env)
    env = VecEnv(env)
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
        "EPOCHS": 10000,
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
    _rollout = make_rollout(rollout_config, env, env_params, training=False)
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
        run={"num_rollouts": num_rollouts, "discrete": discrete, "action_dim": action_dim},
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

        batch = jax.tree_util.tree_map(lambda x: x[:pointer], dataset)

        # TRAIN MODEL
        history = train_model(
            model,
            optimizer,
            metrics,
            batch,
            model_config["EPOCHS"],
            pointer,
            model_config["MINIBATCH_SIZE"],
            rngs=rngs,
        )

        logger.log_loss_history(history, j)

        # PLOT TRUE VS PREDICTED DYNAMICS & REWARDS
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
            )
            logger.log_validation_metrics(dyn_mae, rew_mae, term_bce, term_f1, j)

        # PLOT UNCERTAINTY & MEAN PREDICTIONS (heatmap over state space)
        rng, _rng = jax.random.split(rng)
        plotting.evaluate_and_plot_uncertainty(
            model, base_env, env_params, env_name, _rng, dataset, pointer, j, log_dir
        )

        # Train model-env explore policy
        model_env_explore = ModelEnvironment(
            env,
            env_params,
            model,
            alpha=args.alpha,
            beta=args.beta,
            prediction_mode=args.model_env_mode,
        )
        model_env_explore_params = model_env_explore.default_params
        model_env_explore = LogWrapper(model_env_explore)
        if not discrete:
            model_env_explore = ClipAction(model_env_explore)
        model_env_explore = VecEnv(model_env_explore)

        train_jit = nnx.jit(
            make_train(model_env_explore, model_env_explore_params, config)
        )
        train_state = make_train_state(
            config, model_env_explore, model_env_explore_params, rngs
        )
        rng, _rng = jax.random.split(rng)
        out = train_jit(train_state, _rng)
        train_state = out["runner_state"][0]
        runner_state = (
            train_state,
            runner_state[1],
            runner_state[2],
            runner_state[3],
        )

        returned_episode = out["metrics"]["returned_episode"]
        timesteps = out["metrics"]["timestep"][returned_episode] * config["NUM_ENVS"]
        returns = out["metrics"]["returned_episode_returns"][returned_episode]

        logger.log_ppo_returns(timesteps, returns, "ppo/explore_return")

        # Train model-env eval policy
        model_env_eval = ModelEnvironment(
            env,
            env_params,
            model,
            alpha=1.0,
            beta=0.0,
            prediction_mode=args.model_env_mode,
        )
        model_env_eval_params = model_env_eval.default_params
        model_env_eval = LogWrapper(model_env_eval)
        if not discrete:
            model_env_eval = ClipAction(model_env_eval)
        model_env_eval = VecEnv(model_env_eval)

        train_jit = nnx.jit(make_train(model_env_eval, model_env_eval_params, config))
        eval_train_state = make_train_state(
            config, model_env_eval, model_env_eval_params, rngs
        )
        rng, _rng = jax.random.split(rng)
        out = train_jit(eval_train_state, _rng)
        eval_train_state = out["runner_state"][0]

        returned_episode = out["metrics"]["returned_episode"]
        timesteps = out["metrics"]["timestep"][returned_episode] * config["NUM_ENVS"]
        returns = out["metrics"]["returned_episode_returns"][returned_episode]

        logger.log_ppo_returns(timesteps, returns, "ppo/eval_return")

        rng, _rng = jax.random.split(rng)
        mean_return = evaluation.evaluate_policy(
            eval_config, env, env_params, eval_train_state, _rng, make_rollout
        )
        logger.log_eval_return(pointer, mean_return)

    logger.close()


if __name__ == "__main__":
    main()
