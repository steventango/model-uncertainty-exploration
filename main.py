import dataclasses
import json
import os
import time
from datetime import datetime
from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import tyro
from flax import nnx
from gymnax.environments import spaces

import evaluation
import plotting
import sweep as sweep_lib
import validation
from config import Args, model_config_dict, ppo_config_dict
from data import collate_rollout
from environments import BRAX_BACKENDS, BRAX_ENVS, make_state_reconstruction_wrapper
from logger import ExperimentLogger, log_eval, log_validation
from model_env import ModelEnvironment, reset_weights
from models import make_batched_model, make_batched_rngs, make_batched_train_model
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
from wrappers import BraxGymnaxWrapper, ClipAction, LogWrapper, VecEnv


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
    args = tyro.cli(Args)

    B = args.num_seeds
    seeds = args.seed + jnp.arange(B)
    seed_keys = jax.vmap(jax.random.key)(seeds)

    if args.offline:
        print(f"Loading offline dataset: {args.dataset}")
        all_transitions, action_space, observation_space, _ = (
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
        # Single source of truth for the dataset buffer size; the RBF feature
        # bank places one center per stored datapoint, so NUM_FEATURES (rbf) is
        # derived from this same value (see model_config_dict).
        dataset_size = total_timesteps

        num_rollouts = args.num_rollouts if args.num_rollouts is not None else 1
        log_prefix = f"plant_{args.dataset.replace('/', '_')}"
        hparams_run = {
            "num_rollouts": num_rollouts,
            "N_train": N,
            "action_dim": action_dim,
        }
        hparams_extra = {}

    else:
        is_brax = args.env in BRAX_ENVS
        if is_brax:
            backend = BRAX_BACKENDS.get(args.env, "positional")
            base_env = BraxGymnaxWrapper(args.env, backend=backend)
            env_params = base_env.default_params
        else:
            base_env, env_params = gymnax.make(args.env)

        if is_brax:
            predict_reward_terminated = args.predict_reward_terminated or not hasattr(
                base_env, "obs_to_reward_terminated"
            )
        else:
            predict_reward_terminated = args.predict_reward_terminated
        oracle_reward_terminated = not predict_reward_terminated
        # The state-reconstruction wrapper is only needed when the model env uses the
        # oracle for reward/termination (ModelEnvironment.get_state is its sole caller,
        # gated on oracle_reward_terminated). Skip it when --predict_reward_terminated is
        # set so envs without a registered wrapper don't hit the wrapper's ValueError.
        if not predict_reward_terminated:
            base_env = make_state_reconstruction_wrapper(base_env, args.env)
        action_space = base_env.action_space(env_params)
        discrete = isinstance(action_space, spaces.Discrete)
        action_dim = action_space.n if discrete else action_space.shape[0]
        act_dim = action_space.n if discrete else None
        env = _wrap_env(base_env, discrete)

        env_name = args.env
        if is_brax:
            (
                val_obs,
                val_act,
                val_true_delta_obs,
                val_true_reward,
                val_true_terminated,
            ) = validation.generate_brax_validation_data(base_env, env_params)
        else:
            (
                val_obs,
                val_act,
                val_true_delta_obs,
                val_true_reward,
                val_true_terminated,
            ) = validation.generate_validation_data(base_env, env_params, env_name)

        obs_dim = env.observation_space(env_params).shape[0]
        in_features = obs_dim + action_dim
        out_features = obs_dim + 2 if predict_reward_terminated else obs_dim

        rollout_steps = env_params.max_steps_in_episode # // 10
        num_steps = 10
        model_minibatch_size = rollout_steps
        # Floor of the episode count: aim for roughly max_steps env-steps of data,
        # but this is NOT a step-count guarantee. Integer division never rounds up
        # and the max(1, ...) floor means a single episode for envs with
        # max_steps_in_episode in the upper half of the range.
        max_steps = 100_000 if is_brax else 1000
        num_episodes = max(1, max_steps // env_params.max_steps_in_episode)
        total_timesteps = env_params.max_steps_in_episode * num_episodes
        # Cap the rolling dataset buffer; the RBF feature bank places one center
        # per stored datapoint, so NUM_FEATURES (rbf) is derived from this same
        # value (see model_config_dict) to keep the two in lockstep.
        dataset_size = min(10000, total_timesteps)

        num_rollouts = args.num_rollouts or int(total_timesteps // rollout_steps)
        log_prefix = env_name
        hparams_run = {
            "num_rollouts": num_rollouts,
            "discrete": discrete,
            "action_dim": action_dim,
        }
        hparams_extra = {}

    sweep_configs = sweep_lib.validate_and_expand(args.model)
    model_config = model_config_dict(
        sweep_configs[0], max_data=dataset_size, minibatch_size=model_minibatch_size
    )
    config = ppo_config_dict(
        args.ppo, env_name=args.env, seed=args.seed, offline=args.offline
    )

    if not args.offline:
        rollout_config = {
            **config,
            "NUM_ENVS": 1,
            "NUM_STEPS": rollout_steps,
            "TOTAL_TIMESTEPS": total_timesteps,
            "DATASET_SIZE": dataset_size,
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

    do_sweep = len(sweep_configs) > 1

    seed_keys, model_keys = vsplit(seed_keys)
    models, train_state = make_batched_model(
        args.model.name,
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
        args.model.name, model_config["UPDATE_STEPS"], model_config["MINIBATCH_SIZE"]
    )
    batched_val_metrics = validation.make_batched_validation_metrics()

    if do_sweep:
        # Per-seed val scorer (both model and data axes batched).
        per_seed_val_metrics = validation.make_per_seed_validation_metrics()
        # Candidate train fn (iterates over C configs, each seed-vmapped).
        candidate_train_fn = sweep_lib.make_candidate_train_fn(
            args.model.name, model_config["UPDATE_STEPS"], model_config["MINIBATCH_SIZE"]
        )
        # Keys for the candidate RNGs — split fresh each rollout inside the loop.
        seed_keys, _sweep_base_key = vsplit(seed_keys)

    dt = datetime.now()
    log_dir = (
        args.log_dir
        or f"runs/{dt.strftime('%Y%m%d')}/{log_prefix}/{dt.strftime('%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as _f:
        json.dump({"env": args.env, "model": args.model.name, "label": args.label}, _f)

    seed_dirs, loggers = [], []
    for b in range(B):
        seed_dir = f"{log_dir}/seed_{int(seeds[b])}"
        logger = ExperimentLogger(seed_dir)
        args_flat = {
            k: v
            for k, v in dataclasses.asdict(args).items()
            if k not in ("ppo", "model")
        }
        args_flat["model"] = args.model.name
        args_flat["seed"] = int(seeds[b])
        logger.log_hparams(
            args=args_flat,
            ppo=config,
            model=model_config,
            run=hparams_run,
            **hparams_extra,
        )
        seed_dirs.append(seed_dir)
        loggers.append(logger)

    if args.offline:
        oracle_reward_terminated = True
    # Offline has no real env to reset; "env" falls back to the dataset's
    # initial-state distribution ("init").
    reset_source = (
        "init" if args.offline and args.reset_source == "env" else args.reset_source
    )
    model_env = ModelEnvironment(
        env,
        env_params,
        prediction_mode=args.model_env_mode,
        explore_bonus=args.explore_bonus,
        oracle_reward_terminated=oracle_reward_terminated,
        reset_source=reset_source,
        max_steps_in_episode=args.rollout_length,
        uncertainty_threshold=args.uncertainty_threshold,
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
        seed_keys, val_keys = vsplit(seed_keys)
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
            val_keys,
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
            t0 = time.perf_counter()
            runner_state, traj_batch = _rollout(runner_state)
            jax.block_until_ready(traj_batch)
            rollout_s = time.perf_counter() - t0

            # UPDATE DATASET
            traj_batch = jax.vmap(collate_rollout)(traj_batch)  # (B, T*E, ...)
            dataset = append_dataset(dataset, traj_batch, jnp.asarray(data_count))
            for b in range(B):
                loggers[b].log_dataset(_seed_slice(traj_batch, b), data_count)
            data_count += traj_batch.obs.shape[1]
            for b in range(B):
                loggers[b].log_scalar("time/rollout_s", rollout_s, data_count)

        # TRAIN MODEL (B seeds in parallel)
        t0 = time.perf_counter()
        if do_sweep:
            train_ptr = int(0.8 * data_count)
            # Build fresh candidates branched from current deployed model's config.
            seed_keys, cand_key = vsplit(seed_keys)
            candidates, sweepable_vals = sweep_lib.make_candidates(
                sweep_configs,
                args.model.name,
                in_features=in_features,
                obs_dim=obs_dim,
                out_features=out_features,
                act_dim=act_dim,
                keys=cand_key,
                max_data=dataset_size,
                minibatch_size=model_minibatch_size,
                predict_reward_terminated=predict_reward_terminated,
            )
            # Build per-candidate RNG streams.
            C = len(candidates)
            seed_keys, cand_rng_seed = vsplit(seed_keys)
            cand_rngs_list = [
                make_batched_rngs(jax.vmap(lambda k, i=c: jax.random.fold_in(k, i))(cand_rng_seed))
                for c in range(C)
            ]
            # Train candidates on first 80% of data (throwaway).
            candidate_train_fn(candidates, dataset, jnp.asarray(train_ptr), cand_rngs_list)
            # Score each candidate on held-out 20%.
            val_obs_h, val_act_h, val_delta_h, val_rew_h, val_term_h = (
                validation.held_out_val_data(dataset, train_ptr, data_count)
            )
            seed_keys, score_keys = vsplit(seed_keys)
            dyn_maes = []
            for c_idx, (m_c, _) in enumerate(candidates):
                dyn_mae_c, _, _, _, _ = per_seed_val_metrics(
                    m_c, val_obs_h, val_act_h, val_delta_h, val_rew_h, val_term_h,
                    score_keys,
                )
                dyn_maes.append(dyn_mae_c)
            scores = jnp.stack(dyn_maes, axis=0)  # (C, B)
            best_c = jnp.argmin(scores, axis=0)   # (B,)
            # Apply winning hyperparameters to the persistent deployed model.
            sweep_lib.apply_winning_hypers(
                models, sweep_configs, best_c, args.model.name, sweepable_vals
            )
            # Log which config each seed chose.
            for b in range(B):
                chosen = int(best_c[b])
                loggers[b].log_scalar("sweep/best_config_idx", chosen, data_count)
                chosen_cfg = sweep_configs[chosen]
                for field in sweep_lib.SWEEPABLE.get(args.model.name, set()):
                    loggers[b].log_scalar(
                        f"sweep/chosen_{field}", float(getattr(chosen_cfg, field)), data_count
                    )
                for c_idx in range(C):
                    loggers[b].log_scalar(
                        f"sweep/dyn_mae_config_{c_idx}", float(scores[c_idx, b]), data_count
                    )

        # Retrain deployed model on full data (warm-started).
        history = batched_train_model(
            models, train_state, dataset, jnp.asarray(data_count), batched_rngs
        )
        jax.block_until_ready(history)
        model_train_s = time.perf_counter() - t0
        for b in range(B):
            loggers[b].log_loss_history(_seed_slice(history, b), j)
            loggers[b].log_scalar("time/model_train_s", model_train_s, data_count)

        if not args.offline:
            seed_keys, val_keys = vsplit(seed_keys)
            t0 = time.perf_counter()
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
                val_keys,
            )
            validation_s = time.perf_counter() - t0
            for b in range(B):
                loggers[b].log_scalar("time/validation_s", validation_s, data_count)

        if not args.offline and args.debug and not is_brax:
            seed_keys, plot_seed = vsplit(seed_keys)
            seed_keys, unc_seed = vsplit(seed_keys)
            t0 = time.perf_counter()
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
            # NOTE: this wraps evaluate_validation (eval compute) AND the
            # uncertainty plotting, so the metric reflects eval + plot time.
            validation_eval_plot_s = time.perf_counter() - t0
            for b in range(B):
                loggers[b].log_scalar(
                    "time/validation_eval_plot_s", validation_eval_plot_s, data_count
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
        if reset_source != "env":
            # Per-seed buffer + reset sampling weights (uniform over the full
            # buffer for "buffer", over episode-start rows for "init").
            weights = reset_weights(
                dataset.terminated, dataset.truncated, data_count, reset_source
            )
            model_env_params_b = model_env_params_b.replace(
                init_obs=dataset.obs, init_weights=weights
            )
        seed_keys, train_seed = vsplit(seed_keys)
        train_rng = vsplit(train_seed, num_configs).T  # (B, C)
        t0 = time.perf_counter()
        out = batched_train_jit(batched_train_state_ppo, model_env_params_b, train_rng)
        jax.block_until_ready(out)
        ppo_train_s = time.perf_counter() - t0
        for b in range(B):
            loggers[b].log_scalar("time/ppo_train_s", ppo_train_s, data_count)
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
        t0 = time.perf_counter()
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
        eval_s = time.perf_counter() - t0
        for b in range(B):
            loggers[b].log_scalar("time/eval_s", eval_s, data_count)

    if num_rollouts > 0:
        t0 = time.perf_counter()
        for b in range(B):
            _save_checkpoint(
                seed_dirs[b],
                models,
                explore_train_state,
                eval_train_state,
                b,
            )
        checkpoint_s = time.perf_counter() - t0
        for b in range(B):
            loggers[b].log_scalar("time/checkpoint_s", checkpoint_s, data_count)

    for logger in loggers:
        logger.close()

    open(os.path.join(log_dir, "COMPLETE"), "w").close()


if __name__ == "__main__":
    main()
