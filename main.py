import argparse
import os
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
from logger import ExperimentLogger
from model import make_batched_model, make_batched_rngs, make_batched_train_model
from model_env import ModelEnvironment
from ppo import (
    make_batched_train,
    make_batched_train_state,
    make_rollout,
    make_train_state,
    select_config_train_state,
    unstack_train_state,
)
from wrappers import ClipAction, LogWrapper, VecEnv

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"


def _wrap_env(env, discrete):
    wrapped = LogWrapper(env)
    if not discrete:
        wrapped = ClipAction(wrapped)
    return VecEnv(wrapped)


def _seed_slice(tree, idx):
    """Index every leaf of a batched pytree, e.g. pick one seed (or (seed, config))."""
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


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
        default="mean",
        choices=["mean", "sample"],
        help="Model env transition mode: mean (base net) or sample (epinet at z[0])",
    )
    parser.add_argument(
        "--explore_bonus",
        type=str,
        default="std",
        choices=["std", "eig"],
        help="Intrinsic exploration bonus: std (epinet std) or eig (½ log(1 + σ²_ep))",
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

    B = args.num_seeds
    seeds = args.seed + jnp.arange(B)
    seed_keys = jax.vmap(jax.random.key)(seeds)  # (B,) one master key per seed

    base_env, env_params = gymnax.make(rollout_config["ENV_NAME"])
    action_space = base_env.action_space(env_params)
    discrete = isinstance(action_space, spaces.Discrete)
    action_dim = action_space.n if discrete else action_space.shape[0]
    act_dim = action_space.n if discrete else None
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

    obs_dim = env.observation_space(env_params).shape[0]
    in_features = obs_dim + action_dim
    out_features = obs_dim + 2

    # INIT B ROLLOUT POLICIES (one per seed)
    seed_keys, policy_keys = vsplit(seed_keys)

    @nnx.vmap(in_axes=0, out_axes=0)
    def _build_rollout_train_state(key):
        return make_train_state(config, env, env_params, nnx.Rngs(params=key))

    batched_rollout_train_state = _build_rollout_train_state(policy_keys)

    # INIT B ENVS
    seed_keys, reset_seed = vsplit(seed_keys)
    reset_rng = vsplit(reset_seed, rollout_config["NUM_ENVS"]).T  # (B, NUM_ENVS)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    # BATCHED ROLLOUT
    _rollout = nnx.jit(
        nnx.vmap(make_rollout(rollout_config, env, env_params, training=False))
    )

    # MOCK ROLLOUT for dataset shapes
    seed_keys, mock_rng = vsplit(seed_keys)
    runner_state = (batched_rollout_train_state, env_state, obsv, mock_rng)
    _, traj_batch = _rollout(runner_state)
    sample = jax.vmap(collate_rollout)(traj_batch)  # (B, T*E, ...)

    # INIT DATASET (B, DATASET_SIZE, ...)
    dataset = jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (B, rollout_config["DATASET_SIZE"]) + x.shape[2:], dtype=x.dtype
        ),
        sample,
    )
    pointer = 0

    @jax.jit
    def append_dataset(dataset, traj_batch, ptr):
        def append(old, new):
            return jax.lax.dynamic_update_slice_in_dim(old, new, ptr, axis=0)

        return jax.tree_util.tree_map(jax.vmap(append), dataset, traj_batch)

    # INIT B WORLD MODELS + OPTIMIZERS + METRICS
    seed_keys, model_keys = vsplit(seed_keys)
    models, optimizers, metrics = make_batched_model(
        model_config, in_features, obs_dim, out_features, act_dim, model_keys
    )
    seed_keys, train_model_seed = vsplit(seed_keys)
    batched_rngs = make_batched_rngs(train_model_seed)
    batched_train_model = make_batched_train_model(
        model_config["UPDATE_STEPS"], model_config["MINIBATCH_SIZE"]
    )
    batched_val_metrics = validation.make_batched_validation_metrics()

    # PER-SEED LOGGERS
    log_dir = args.log_dir or (
        f"runs/{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    num_rollouts = int(
        rollout_config["TOTAL_TIMESTEPS"]
        // rollout_config["NUM_STEPS"]
        // rollout_config["NUM_ENVS"]
    )
    seed_dirs, loggers = [], []
    for b in range(B):
        seed_dir = f"{log_dir}/seed_{int(seeds[b])}"
        logger = ExperimentLogger(seed_dir)
        logger.log_hparams(
            args={**vars(args), "seed": int(seeds[b])},
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
        seed_dirs.append(seed_dir)
        loggers.append(logger)

    model_env = ModelEnvironment(
        env,
        env_params,
        prediction_mode=args.model_env_mode,
        explore_bonus=args.explore_bonus,
    )
    model_env = _wrap_env(model_env, discrete)
    EXPLORE, EVAL = 0, 1
    alphas = jnp.array([args.alpha, 1.0])
    betas = jnp.array([args.beta, 0.0])
    num_configs = alphas.shape[0]
    batched_train_jit = nnx.jit(
        make_batched_train(model_env, model_env.default_params, config)
    )
    batched_eval = evaluation.make_batched_evaluate_policy(
        eval_config,
        env,
        env_params,
        make_rollout(eval_config, env, env_params, training=False),
    )

    seed_keys, runner_seed = vsplit(seed_keys)
    runner_state = (batched_rollout_train_state, env_state, obsv, runner_seed)

    for j in range(num_rollouts):
        # ROLLOUT (B seeds in parallel)
        runner_state, traj_batch = _rollout(runner_state)

        # UPDATE DATASET
        traj_batch = jax.vmap(collate_rollout)(traj_batch)  # (B, T*E, ...)
        dataset = append_dataset(dataset, traj_batch, jnp.asarray(pointer))
        for b in range(B):
            loggers[b].log_dataset(_seed_slice(traj_batch, b), pointer)
        pointer += traj_batch.obs.shape[1]

        # TRAIN MODEL (B seeds in parallel)
        history = batched_train_model(
            models, optimizers, metrics, dataset, jnp.asarray(pointer), batched_rngs
        )
        for b in range(B):
            loggers[b].log_loss_history(_seed_slice(history, b), j)

        if pointer > 0:
            dyn_mae, rew_mae, term_bce, term_f1 = batched_val_metrics(
                models,
                val_obs,
                val_act,
                val_true_delta_obs,
                val_true_reward,
                val_true_terminated,
            )
            for b in range(B):
                loggers[b].log_validation_metrics(
                    dyn_mae[b], rew_mae[b], term_bce[b], term_f1[b], j
                )
            if config["DEBUG"]:
                seed_keys, plot_seed = vsplit(seed_keys)
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
                    pointer,
                    plot_seed[0],
                    j,
                    seed_dirs[0],
                    plot=True,
                )

        if config["DEBUG"]:
            seed_keys, unc_seed = vsplit(seed_keys)
            plotting.evaluate_and_plot_uncertainty(
                unstack_train_state(models, 0),
                base_env,
                env_params,
                env_name,
                unc_seed[0],
                _seed_slice(dataset, 0),
                pointer,
                j,
                seed_dirs[0],
            )

        # BATCHED PPO TRAIN (B seeds x C configs)
        seed_keys, ts_seed = vsplit(seed_keys)
        ts_keys = vsplit(ts_seed, num_configs).T  # (B, C)
        batched_train_state = make_batched_train_state(
            config, model_env, model_env.default_params, ts_keys
        )
        # model varies per seed (B axis); alpha/beta vary per config (C axis).
        model_env_params_b = model_env.default_params.replace(
            model=models, alpha=alphas, beta=betas
        )
        seed_keys, train_seed = vsplit(seed_keys)
        train_rng = vsplit(train_seed, num_configs).T  # (B, C)
        out = batched_train_jit(batched_train_state, model_env_params_b, train_rng)
        batched_out_train_state = out["runner_state"][0]
        explore_train_state = select_config_train_state(
            batched_out_train_state, EXPLORE
        )
        eval_train_state = select_config_train_state(batched_out_train_state, EVAL)

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
        mean_returns = batched_eval(eval_train_state, eval_keys)  # (B,)
        for b in range(B):
            loggers[b].log_eval_return(pointer, mean_returns[b])

    for logger in loggers:
        logger.close()

    # Completion sentinel for resubmission tooling (slurm/vulcan/grid_tasks.py).
    open(os.path.join(log_dir, "COMPLETE"), "w").close()


if __name__ == "__main__":
    main()
