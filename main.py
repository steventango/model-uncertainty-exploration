import argparse
import os
import orbax.checkpoint as ocp
from datetime import datetime
from functools import partial

import gymnax
import jax

import jax.numpy as jnp
from flax import nnx
from gymnax.environments import spaces

import evaluation
import plotting
import validation
from data import collate_rollout
from env_config import SUPPORTED_ENVS
from logger import ExperimentLogger, log_eval, log_validation
from models import make_batched_model, make_batched_rngs, make_batched_train_model
from model_env import ModelEnvironment
from offline_data import AREA_INDEX, load_offline_transitions
from plant_env import PlantEnv, PlantEnvParams
from ppo import (
    make_batched_train,
    make_batched_train_state,
    make_rollout,
    make_train_state,
    select_config_train_state,
    unstack_train_state,
)
from environments import make_state_reconstruction_wrapper
from wrappers import ClipAction, LogWrapper, VecEnv


def _wrap_env(env, discrete):
    wrapped = LogWrapper(env)
    if not discrete:
        wrapped = ClipAction(wrapped)
    return VecEnv(wrapped)


def _seed_slice(tree, idx):
    """Index every leaf of a batched pytree, e.g. pick one seed (or (seed, config))."""
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def _save_checkpoint(seed_dir, models, explore_train_state, eval_train_state, seed_idx):
    checkpointer = ocp.StandardCheckpointer()
    abs_seed_dir = os.path.abspath(seed_dir)

    _, model_state = nnx.split(unstack_train_state(models, seed_idx))
    checkpointer.save(os.path.join(abs_seed_dir, "checkpoint", "model"), model_state)

    for tag, batched_ts in [
        ("ppo_explore", explore_train_state),
        ("ppo_eval", eval_train_state),
    ]:
        network, _, normalize_vec_obs, _ = unstack_train_state(batched_ts, seed_idx)
        _, network_state = nnx.split(network)
        _, obs_norm_state = nnx.split(normalize_vec_obs)
        base = os.path.join(abs_seed_dir, "checkpoint", tag)
        checkpointer.save(os.path.join(base, "network"), network_state)
        checkpointer.save(os.path.join(base, "obs_norm"), obs_norm_state)

    checkpointer.wait_until_finished()


@partial(jax.jit, static_argnums=1)
def vsplit(keys, num=2):
    return jax.vmap(lambda k: jax.random.split(k, num), out_axes=1)(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0, help="Exploit weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Explore weight")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help="Number of seeds to train concurrently (vmapped on one GPU)",
    )
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
        default="sample",
        choices=["mean", "sample"],
        help="Model env transition mode: mean (base net) or sample (epinet at z[0])",
    )
    parser.add_argument(
        "--explore_bonus",
        type=str,
        default="eig",
        choices=["std", "eig"],
        help="Intrinsic exploration bonus: std (epinet std) or eig (½ log(1 + σ²_ep))",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug plots (true vs predicted, uncertainty heatmaps)",
    )
    parser.add_argument(
        "--predict_reward_terminated",
        action="store_true",
        help="Use model-predicted reward/terminated instead of reward/terminated from the real env.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=None,
        help="Number of rollout iterations.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="enn",
        choices=["enn", "blr"],
        help="World model type.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode: load a minari dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="plant-data/visu-v27",
        help="Minari dataset id for offline mode.",
    )
    args = parser.parse_args()

    B = args.num_seeds
    seeds = args.seed + jnp.arange(B)
    seed_keys = jax.vmap(jax.random.key)(seeds)

    if args.offline:
        print(f"Loading offline dataset: {args.dataset}")
        all_transitions, action_space, observation_space, init_areas, _ = (
            load_offline_transitions(args.dataset)
        )
        action_dim = action_space.shape[0]
        act_dim = None  # continuous action — no one-hot encoding
        N = all_transitions.obs.shape[0]
        print(f"  N={N} | action_dim={action_dim}")

        env = PlantEnv(action_dim)
        env_params = PlantEnvParams(
            area_min=float(observation_space.low[AREA_INDEX]),
            area_max=float(observation_space.high[AREA_INDEX]),
            act_low=jnp.asarray(action_space.low, dtype=jnp.float32),
            act_high=jnp.asarray(action_space.high, dtype=jnp.float32),
            init_areas=init_areas,
            max_steps_in_episode=14,
        )

        dataset = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (B,) + x.shape).copy(),
            all_transitions,
        )

        obs_dim = 1
        in_features = obs_dim + action_dim
        out_features = obs_dim
        predict_reward_terminated = False
        discrete = False

        num_steps = 14
        model_minibatch_size = min(N, 256)
        total_timesteps = N

        num_rollouts = args.num_rollouts if args.num_rollouts is not None else 1
        log_prefix = f"plant_{args.dataset.replace('/', '_')}"
        hparams_run = {
            "num_rollouts": num_rollouts,
            "N_train": N,
            "action_dim": action_dim,
        }
        hparams_extra = {}

    else:
        base_env, env_params = gymnax.make(args.env)
        # The state-reconstruction wrapper is only needed when the model env uses the
        # oracle for reward/termination (ModelEnvironment.get_state is its sole caller,
        # gated on oracle_reward_terminated). Skip it when --predict_reward_terminated is
        # set so envs without a registered wrapper don't hit the wrapper's ValueError.
        if not args.predict_reward_terminated:
            base_env = make_state_reconstruction_wrapper(base_env, args.env)
        action_space = base_env.action_space(env_params)
        discrete = isinstance(action_space, spaces.Discrete)
        action_dim = action_space.n if discrete else action_space.shape[0]
        act_dim = action_space.n if discrete else None
        env = _wrap_env(base_env, discrete)

        env_name = args.env
        (
            val_obs,
            val_act,
            val_true_delta_obs,
            val_true_reward,
            val_true_terminated,
        ) = validation.generate_validation_data(base_env, env_params, env_name)

        obs_dim = env.observation_space(env_params).shape[0]
        in_features = obs_dim + action_dim
        out_features = obs_dim + 2 if args.predict_reward_terminated else obs_dim
        predict_reward_terminated = args.predict_reward_terminated

        rollout_steps = env_params.max_steps_in_episode // 10
        num_steps = 10
        model_minibatch_size = rollout_steps
        num_episodes = 5
        total_timesteps = env_params.max_steps_in_episode * num_episodes

        num_rollouts = args.num_rollouts or int(
            total_timesteps // rollout_steps
        )
        log_prefix = env_name
        hparams_run = {
            "num_rollouts": num_rollouts,
            "discrete": discrete,
            "action_dim": action_dim,
        }
        hparams_extra = {}
    if args.model == "blr":
        model_config = {
            "MAX_DATA": max_data,
            "NUM_SAMPLES": 10,
            "LAM": 0.01,
            "A0": 1.0,
            "B0": 1.0,
            "LENGTH_SCALE": 1.0,
            "UPDATE_STEPS": 1,
            "MINIBATCH_SIZE": model_minibatch_size,
        }
    else:
        model_config = {
            "LR": 1e-3,
            "HIDDEN_DIM": 64,
            "LEARNABLE_HIDDEN_DIM": 15,
            "PRIOR_HIDDEN_DIM": 5,
            "INDEX_DIM": 8,
            "ACTIVATION": "tanh",
            "UPDATE_STEPS": 10000,
            "MINIBATCH_SIZE": model_minibatch_size,
        }

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
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "SEED": args.seed,
        "DEBUG": False,
    }
    if not args.offline:
        config["ENV_NAME"] = args.env
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if not args.offline:
        rollout_config = {
            **config,
            "NUM_ENVS": 1,
            "NUM_STEPS": rollout_steps,
            "TOTAL_TIMESTEPS": total_timesteps,
            "DATASET_SIZE": min(10000, total_timesteps),
        }
        eval_config = {
            **rollout_config,
            "NUM_ENVS": 100,
            "NUM_STEPS": env_params.max_steps_in_episode,
        }
        hparams_extra = {"rollout": rollout_config, "eval": eval_config}

        seed_keys, policy_keys = vsplit(seed_keys)

        @nnx.vmap(in_axes=0, out_axes=0)
        def _build_rollout_train_state(key):
            return make_train_state(config, env, env_params, nnx.Rngs(params=key))

        batched_rollout_train_state = _build_rollout_train_state(policy_keys)

        seed_keys, reset_seed = vsplit(seed_keys)
        reset_rng = vsplit(reset_seed, rollout_config["NUM_ENVS"]).T  # (B, NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        _rollout = nnx.jit(
            nnx.vmap(make_rollout(rollout_config, env, env_params, training=False))
        )

        seed_keys, mock_rng = vsplit(seed_keys)
        _, traj_batch = _rollout(
            (batched_rollout_train_state, env_state, obsv, mock_rng)
        )
        sample = jax.vmap(collate_rollout)(traj_batch)  # (B, T*E, ...)

        dataset = jax.tree_util.tree_map(
            lambda x: jnp.zeros(
                (B, rollout_config["DATASET_SIZE"]) + x.shape[2:], dtype=x.dtype
            ),
            sample,
        )
        data_count = 0

        @jax.jit
        def append_dataset(dataset, traj_batch, ptr):
            def append(old, new):
                return jax.lax.dynamic_update_slice_in_dim(old, new, ptr, axis=0)

            return jax.tree_util.tree_map(jax.vmap(append), dataset, traj_batch)

    seed_keys, model_keys = vsplit(seed_keys)
    models, train_state = make_batched_model(
        args.model,
        model_config,
        in_features,
        obs_dim,
        out_features,
        act_dim,
        model_keys,
        predict_reward_terminated=predict_reward_terminated,
    )
    seed_keys, train_model_seed = vsplit(seed_keys)
    batched_rngs = make_batched_rngs(train_model_seed)
    batched_train_model = make_batched_train_model(
        args.model, model_config["UPDATE_STEPS"], model_config["MINIBATCH_SIZE"]
    )
    batched_val_metrics = validation.make_batched_validation_metrics()

    log_dir = (
        args.log_dir or f"runs/{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    seed_dirs, loggers = [], []
    for b in range(B):
        seed_dir = f"{log_dir}/seed_{int(seeds[b])}"
        logger = ExperimentLogger(seed_dir)
        logger.log_hparams(
            args={**vars(args), "seed": int(seeds[b])},
            ppo=config,
            model=model_config,
            run=hparams_run,
            **hparams_extra,
        )
        seed_dirs.append(seed_dir)
        loggers.append(logger)

    oracle_reward_terminated = (
        True if args.offline else not args.predict_reward_terminated
    )
    model_env = ModelEnvironment(
        env,
        env_params,
        prediction_mode=args.model_env_mode,
        explore_bonus=args.explore_bonus,
        oracle_reward_terminated=oracle_reward_terminated,
    )
    if not args.offline:
        batched_eval = evaluation.make_batched_evaluate_policy(
            eval_config,
            env,
            env_params,
            make_rollout(eval_config, env, env_params, training=False),
        )
    model_env = _wrap_env(model_env, discrete)

    EXPLORE, EVAL = 0, 1
    alphas = jnp.array([args.alpha, 1.0])
    betas = jnp.array([args.beta, 0.0])
    num_configs = alphas.shape[0]
    batched_train_jit = nnx.jit(
        make_batched_train(model_env, model_env.default_params, config)
    )
    if not args.offline:
        log_validation(
            loggers,
            batched_val_metrics,
            models,
            val_obs,
            val_act,
            val_true_delta_obs,
            val_true_reward,
            val_true_terminated,
            0,
        )
        seed_keys, runner_seed = vsplit(seed_keys)
        runner_state = (batched_rollout_train_state, env_state, obsv, runner_seed)
        seed_keys, eval_keys = vsplit(seed_keys)
        log_eval(loggers, batched_eval, batched_rollout_train_state, eval_keys, 0)

    if args.offline:
        data_count = N

    for j in range(num_rollouts):
        if not args.offline:
            # ROLLOUT (B seeds in parallel)
            runner_state, traj_batch = _rollout(runner_state)

            # UPDATE DATASET
            traj_batch = jax.vmap(collate_rollout)(traj_batch)  # (B, T*E, ...)
            dataset = append_dataset(dataset, traj_batch, jnp.asarray(data_count))
            for b in range(B):
                loggers[b].log_dataset(_seed_slice(traj_batch, b), data_count)
            data_count += traj_batch.obs.shape[1]

        # TRAIN MODEL (B seeds in parallel)
        history = batched_train_model(
            models, train_state, dataset, jnp.asarray(data_count), batched_rngs
        )
        for b in range(B):
            loggers[b].log_loss_history(_seed_slice(history, b), j)

        if not args.offline:
            log_validation(
                loggers,
                batched_val_metrics,
                models,
                val_obs,
                val_act,
                val_true_delta_obs,
                val_true_reward,
                val_true_terminated,
                data_count,
            )

        if not args.offline and args.debug:
            seed_keys, plot_seed = vsplit(seed_keys)
            # DEBUG-only diagnostic: plots seed 0 alone (models/dataset/dir
            # are all sliced to seed 0), not all B seeds.
            validation.evaluate_validation(
                unstack_train_state(models, 0),
                base_env,
                env_params,
                env_name,
                val_obs,
                val_act,
                val_true_delta_obs,
                val_true_reward,
                val_true_terminated,
                _seed_slice(dataset, 0),
                data_count,
                plot_seed[0],
                j,
                seed_dirs[0],
                plot=True,
            )

        if not args.offline and args.debug:
            seed_keys, unc_seed = vsplit(seed_keys)
            # DEBUG-only diagnostic: plots seed 0 alone (models/dataset/dir are
            # all sliced to seed 0), not all B seeds.
            plotting.evaluate_and_plot_uncertainty(
                unstack_train_state(models, 0),
                base_env,
                env_params,
                env_name,
                unc_seed[0],
                _seed_slice(dataset, 0),
                data_count,
                j,
                seed_dirs[0],
            )

        # BATCHED PPO TRAIN (B seeds x C configs)
        seed_keys, ts_seed = vsplit(seed_keys)
        ts_keys = vsplit(ts_seed, num_configs).T  # (B, C)
        batched_train_state_ppo = make_batched_train_state(
            config, model_env, model_env.default_params, ts_keys
        )
        # model varies per seed (B axis); alpha/beta vary per config (C axis).
        model_env_params_b = model_env.default_params.replace(
            model=models, alpha=alphas, beta=betas
        )
        seed_keys, train_seed = vsplit(seed_keys)
        train_rng = vsplit(train_seed, num_configs).T  # (B, C)
        out = batched_train_jit(batched_train_state_ppo, model_env_params_b, train_rng)
        batched_out_train_state = out["runner_state"][0]
        explore_train_state = select_config_train_state(
            batched_out_train_state, EXPLORE
        )
        eval_train_state = select_config_train_state(batched_out_train_state, EVAL)

        if not args.offline:
            runner_state = (
                explore_train_state,
                runner_state[1],
                runner_state[2],
                runner_state[3],
            )

        for b in range(B):
            for tag, index in (
                ("ppo/explore_return", EXPLORE),
                ("ppo/eval_return", EVAL),
            ):
                loggers[b].log_ppo_returns(
                    _seed_slice(out["metrics"], (b, index)),
                    tag,
                    j,
                    config["NUM_ENVS"],
                    config["NUM_STEPS"],
                    int(config["TOTAL_TIMESTEPS"]),
                )

        seed_keys, eval_keys = vsplit(seed_keys)
        if not args.offline:
            log_eval(loggers, batched_eval, eval_train_state, eval_keys, data_count)
        else:
            plotting.plot_offline_visualization(
                models,
                explore_train_state,
                eval_train_state,
                model_env,
                config,
                num_steps,
                rng=eval_keys[0],
                dataset=_seed_slice(dataset, 0),
                env_params=env_params,
                j=j,
                run_dir=seed_dirs[0],
                explore_bonus=args.explore_bonus,
                reward_fn=env.compute_reward,
            )

    if num_rollouts > 0:
        for b in range(B):
            _save_checkpoint(
                seed_dirs[b],
                models,
                explore_train_state,
                eval_train_state,
                b,
            )

    for logger in loggers:
        logger.close()

    open(os.path.join(log_dir, "COMPLETE"), "w").close()


if __name__ == "__main__":
    main()
