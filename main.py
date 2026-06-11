import argparse
import os

from flax import nnx
import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
import optax
import pandas as pd

import evaluation
from data import collate_rollout
from env_config import SUPPORTED_ENVS
from model import DynamicsModel, train_model
from model_env import ModelEnvironment
from networks import ENN
import plotting
from ppo_continuous_action import make_rollout, make_train, make_train_state
import validation
from wrappers import ClipAction, LogWrapper, VecEnv

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0, help="Exploit weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Explore weight")
    parser.add_argument(
        "--output_csv", type=str, default="/tmp/metrics.csv", help="Output metrics path"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        choices=SUPPORTED_ENVS,
        help="Gymnax environment name",
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
        "DEBUG": False,
    }
    rollout_config = config.copy()
    rollout_config["NUM_ENVS"] = 1
    rollout_config["DATASET_SIZE"] = 10000
    eval_config = rollout_config.copy()
    eval_config["NUM_ENVS"] = 100

    rng = jax.random.key(30)

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
    real_steps = []
    eval_returns = []
    dyn_maes = []
    rew_maes = []

    num_rollouts = int(
        rollout_config["TOTAL_TIMESTEPS"]
        // rollout_config["NUM_STEPS"]
        // rollout_config["NUM_ENVS"]
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
        pointer += traj_batch.obs.shape[0]

        # PLOT DATASET (OBS, ACTIONS, REWARDS)
        plotting.plot_dataset(
            dataset, pointer, env_name, rollout_config, j, discrete=discrete
        )

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

        # PLOT LOSSES
        plotting.plot_losses(history)

        # PLOT TRUE VS PREDICTED DYNAMICS & REWARDS
        if pointer > 0:
            dyn_mae, rew_mae, term_mae, rng = validation.evaluate_validation(
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
                rng,
                j,
            )
            dyn_maes.append(dyn_mae)
            rew_maes.append(rew_mae)

        # PLOT UNCERTAINTY & MEAN PREDICTIONS (heatmap over state space)
        rng = plotting.evaluate_and_plot_uncertainty(
            model, base_env, env_params, env_name, rng, dataset, pointer, j
        )

        # Train model-env explore policy
        model_env_explore = ModelEnvironment(
            env, env_params, model, alpha=args.alpha, beta=args.beta
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

        plotting.plot_training_curve(
            timesteps,
            returns,
            f"PPO(explore) on Model({env_name})",
            "/tmp/ppo_explore_continuous_action.png",
        )

        # Train model-env eval policy
        model_env_eval = ModelEnvironment(env, env_params, model, alpha=1.0, beta=0.0)
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

        plotting.plot_training_curve(
            timesteps,
            returns,
            f"PPO(eval) on Model({env_name})",
            "/tmp/ppo_eval_continuous_action.png",
        )

        rng, _rng = jax.random.split(rng)
        mean_return = evaluation.evaluate_policy(
            eval_config, env, env_params, eval_train_state, _rng, make_rollout
        )
        real_steps.append(pointer)
        eval_returns.append(mean_return)

        plotting.plot_eval_returns(
            real_steps,
            eval_returns,
            env_name,
            "/tmp/ppo_continuous_action_eval_returns.png",
        )

    # Save validation errors to CSV
    df_metrics = pd.DataFrame(
        {
            "Iteration": list(range(len(dyn_maes))),
            "Dynamics_MAE": dyn_maes,
            "Reward_MAE": rew_maes,
        }
    )
    df_metrics.to_csv(args.output_csv, index=False)
    print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
