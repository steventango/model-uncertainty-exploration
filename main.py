import dataclasses
import json
import os
import time
from datetime import datetime

import gymnax
import jax
import jax.numpy as jnp
import tyro
from flax import nnx
from gymnax.environments import spaces

import evaluation
import plotting
import validation
from agents import Experience, ModelBasedAgent, ModelFreeAgent
from batching import seed_slice, vsplit
from config import Args, model_config_dict, ppo_config_dict
from data import collate_rollout
from environments import BRAX_BACKENDS, BRAX_ENVS, make_state_reconstruction_wrapper
from environments.bsuite.deep_sea import KeyedDeepSea
from logger import ExperimentLogger, log_eval, log_time
from model_env import ModelEnvironment
from offline_data import AREA_INDEX, load_offline_transitions
from plant_env import PlantEnv, PlantEnvParams
from ppo import make_rollout
from wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    OneHotObservationWrapper,
    VecEnv,
)


def _wrap_env(env, discrete):
    wrapped = LogWrapper(env)
    if not discrete:
        wrapped = ClipAction(wrapped)
    return VecEnv(wrapped)


def run_online(
    agent,
    env,
    env_params,
    env_params_axis,
    rollout_config,
    eval_config,
    loggers,
    seed_keys,
    num_rollouts,
    max_steps,
    eval_every,
):
    """Each iteration: act in the real env, add to the buffer, update the agent."""
    key, init_key, reset_key, runner_key = vsplit(seed_keys, 4)
    reset_rng = vsplit(reset_key, rollout_config["NUM_ENVS"]).T  # (B, NUM_ENVS)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, env_params_axis))(
        reset_rng, env_params
    )
    agent_state = agent.init(init_key)

    if agent.on_policy and rollout_config["NORMALIZE_ENV"]:

        @nnx.jit
        @nnx.vmap
        def _init_obs_stats(train_state, obs):
            normalize_vec_obs = train_state[2]
            normalize_vec_obs.train()
            normalize_vec_obs(obs)
            normalize_vec_obs.eval()

        _init_obs_stats(agent.explore_policy(agent_state), obsv)

    def rollout(runner_state, params):
        params = env_params if env_params_axis is None else params
        return make_rollout(rollout_config, env, params, training=agent.on_policy)(
            runner_state
        )

    _rollout = nnx.jit(nnx.vmap(rollout, in_axes=(0, env_params_axis)))
    runner_state = (env_state, obsv, runner_key)

    sample = nnx.eval_shape(
        lambda policy: jax.vmap(collate_rollout)(
            _rollout((policy, *runner_state), env_params)[1]
        ),
        agent.explore_policy(agent_state),
    )
    buffer = jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (x.shape[0], rollout_config["DATASET_SIZE"]) + x.shape[2:], dtype=x.dtype
        ),
        sample,
    )
    size = 0

    @jax.jit
    def append(buffer, batch, ptr):
        def append_one(old, new):
            return jax.lax.dynamic_update_slice_in_dim(old, new, ptr, axis=0)

        return jax.tree_util.tree_map(jax.vmap(append_one), buffer, batch)

    batched_eval = evaluation.make_batched_evaluate_policy(
        eval_config, env, env_params, env_params_axis
    )
    key, eval_keys = vsplit(key)
    log_eval(loggers, batched_eval, agent.eval_policy(agent_state), eval_keys, 0)
    eval_policy = agent.eval_policy(agent_state)

    for rollout in range(num_rollouts):
        t0 = time.perf_counter()
        (_, *runner_state), traj_batch = _rollout(
            (agent.explore_policy(agent_state), *runner_state), env_params
        )
        jax.block_until_ready(traj_batch)
        rollout_s = time.perf_counter() - t0

        collated = jax.vmap(collate_rollout)(traj_batch)  # (B, T*E, ...)
        buffer = append(buffer, collated, jnp.asarray(size))
        for b, logger in enumerate(loggers):
            logger.log_dataset(seed_slice(collated, b), size)
        prev_size, size = size, size + collated.obs.shape[1]
        for logger in loggers:
            logger.log_scalar("time/rollout_s", rollout_s, size)
        # Eval points inside this rollout: the policy has not been updated yet.
        first = (prev_size // eval_every + 1) * eval_every
        for step in range(first, size, eval_every):
            key, eval_keys = vsplit(key)
            log_eval(loggers, batched_eval, eval_policy, eval_keys, step)

        key, update_key = vsplit(key)
        agent_state = agent.update(
            agent_state, Experience(buffer, size, traj_batch, rollout), update_key
        )
        eval_policy = agent.eval_policy(agent_state)

        if size % eval_every == 0 or rollout == num_rollouts - 1:
            key, eval_keys = vsplit(key)
            t0 = time.perf_counter()
            log_eval(loggers, batched_eval, eval_policy, eval_keys, size)
            log_time(loggers, "time/eval_s", t0, size)

    # Grid points past the last rollout: the policy has not changed since.
    first = (size // eval_every + 1) * eval_every
    for step in range(first, max_steps + 1, eval_every):
        key, eval_keys = vsplit(key)
        log_eval(loggers, batched_eval, eval_policy, eval_keys, step)

    return agent_state, size


def run_offline(agent, buffer, loggers, seed_keys, env, env_params, seed_dir, args):
    """One update on the fixed dataset, then plot seed 0."""
    key, init_key = vsplit(seed_keys)
    agent_state = agent.init(init_key)
    size = buffer.obs.shape[1]
    key, update_key = vsplit(key)
    agent_state = agent.update(
        agent_state, Experience(buffer, size, None, 0), update_key
    )

    key, eval_keys = vsplit(key)
    t0 = time.perf_counter()
    plotting.plot_offline_visualization(
        agent_state.models,
        agent_state.explore,
        agent_state.eval,
        agent.model_env,
        agent.ppo_config,
        env_params.max_steps_in_episode,
        rng=eval_keys[0],
        dataset=seed_slice(buffer, 0),
        env_params=env_params,
        j=0,
        run_dir=seed_dir,
        explore_bonus=args.explore_bonus,
        reward_fn=env.compute_reward,
    )
    log_time(loggers, "time/eval_s", t0, size)
    return agent_state, size


def main(args: Args):
    os.environ["XLA_FLAGS"] = (
        f"{os.environ.get('XLA_FLAGS', '')}"
        " --xla_gpu_deterministic_ops=true"
        " --xla_gpu_exclude_nondeterministic_ops=true"
    )

    B = args.num_seeds
    seeds = args.seed + jnp.arange(B)
    seed_keys = jax.vmap(jax.random.key)(seeds)

    if args.offline:
        print(f"Loading offline dataset: {args.dataset}")
        all_transitions, action_space, observation_space, _ = load_offline_transitions(
            args.dataset
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

        buffer = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (B,) + x.shape).copy(),
            all_transitions,
        )

        obs_dim = 1
        in_features = obs_dim + action_dim
        out_features = obs_dim
        predict_reward_terminated = False
        oracle_reward_terminated = True
        discrete = False

        model_minibatch_size = min(N, 256)
        # Single source of truth for the dataset buffer size; the RBF feature
        # bank places one center per stored datapoint, so NUM_FEATURES (rbf) is
        # derived from this same value (see model_config_dict).
        dataset_size = N
        log_prefix = f"plant_{args.dataset.replace('/', '_')}"
        hparams_run = {"num_rollouts": 1, "N_train": N, "action_dim": action_dim}
        val_data = None
        debug = False
        discrete_obs = False
        seed_env_params, env_params_axis = env_params, None

    else:
        is_brax = args.env in BRAX_ENVS
        if is_brax:
            backend = BRAX_BACKENDS.get(args.env, "positional")
            base_env = BraxGymnaxWrapper(args.env, backend=backend)
            env_params = base_env.default_params
        elif args.env.startswith("DeepSea-"):
            base_env = OneHotObservationWrapper(
                KeyedDeepSea(int(args.env.split("-")[1]))
            )
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
        if not predict_reward_terminated and not hasattr(
            base_env, "obs_to_reward_terminated"
        ):
            base_env = make_state_reconstruction_wrapper(base_env, args.env)
        action_space = base_env.action_space(env_params)
        discrete = isinstance(action_space, spaces.Discrete)
        action_dim = action_space.n if discrete else action_space.shape[0]
        act_dim = action_space.n if discrete else None
        discrete_obs = getattr(base_env, "discrete_obs", False)
        if args.env.startswith("DeepSea-"):
            seed_env_params = jax.vmap(base_env.params_for_seed)(seeds)
            env_params_axis = 0
        else:
            seed_env_params, env_params_axis = env_params, None
        env = _wrap_env(base_env, discrete)

        if is_brax:
            val_data = validation.generate_brax_validation_data(base_env, env_params)
        elif args.env.startswith("DeepSea-"):
            val_data = None
        else:
            val_data = validation.generate_validation_data(
                base_env, env_params, args.env
            )

        obs_dim = env.observation_space(env_params).shape[0]
        in_features = obs_dim + action_dim
        out_features = obs_dim + 2 if predict_reward_terminated else obs_dim

        # Real-env steps between re-plans: one episode by default, 1 to re-plan
        # after every real step.
        rollout_steps = args.steps_per_rollout or env_params.max_steps_in_episode
        eval_every = args.eval_every or rollout_steps
        # Tied to episode length so plan-every-step does not shrink it to 1.
        model_minibatch_size = env_params.max_steps_in_episode
        max_steps = args.max_steps or (100_000 if is_brax else 1000)
        num_rollouts = max_steps // rollout_steps
        total_timesteps = num_rollouts * rollout_steps
        # Cap the rolling dataset buffer; the RBF feature bank places one center
        # per stored datapoint, so NUM_FEATURES (rbf) is derived from this same
        # value (see model_config_dict) to keep the two in lockstep.
        dataset_size = (
            min(10000, total_timesteps) if args.model.name == "blr" else total_timesteps
        )
        log_prefix = args.env
        hparams_run = {
            "num_rollouts": num_rollouts,
            "discrete": discrete,
            "action_dim": action_dim,
        }
        debug = args.debug and not is_brax

    model_config = model_config_dict(
        args.model, max_data=dataset_size, minibatch_size=model_minibatch_size
    )
    config = ppo_config_dict(
        args.ppo, env_name=args.env, seed=args.seed, offline=args.offline
    )

    hparams_extra = {}
    rnd_config = args.rnd if args.explore_bonus == "rnd" else None
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
        logger = ExperimentLogger(seed_dir, discrete_obs=discrete_obs)
        args_flat = {
            k: v
            for k, v in dataclasses.asdict(args).items()
            if k not in ("ppo", "model", "rnd")
        }
        args_flat["model"] = args.model.name
        args_flat["seed"] = int(seeds[b])
        logger.log_hparams(
            args=args_flat,
            **{args.solver: config},
            model=model_config,
            run=hparams_run,
            **({"rnd": dataclasses.asdict(rnd_config)} if rnd_config else {}),
            **hparams_extra,
        )
        seed_dirs.append(seed_dir)
        loggers.append(logger)

    if args.model_free:
        agent = ModelFreeAgent(rollout_config, env, env_params, loggers, args.solver)
    else:
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
            seed_env_params=seed_env_params if env_params_axis == 0 else None,
        )
        agent = ModelBasedAgent(
            model_name=args.model.name,
            model_config=model_config,
            model_dims=(in_features, obs_dim, out_features, act_dim),
            predict_reward_terminated=predict_reward_terminated,
            ppo_config=config,
            env=env,
            env_params=env_params,
            model_env=_wrap_env(model_env, discrete),
            reset_source=reset_source,
            alpha=args.alpha,
            beta=args.beta,
            loggers=loggers,
            solver=args.solver,
            rnd_config=rnd_config,
            val_data=val_data,
            debug=(base_env, args.env, seed_dirs[0]) if debug else None,
        )

    if args.offline:
        agent_state, size = run_offline(
            agent, buffer, loggers, seed_keys, env, env_params, seed_dirs[0], args
        )
    else:
        agent_state, size = run_online(
            agent,
            env,
            seed_env_params,
            env_params_axis,
            rollout_config,
            eval_config,
            loggers,
            seed_keys,
            num_rollouts,
            max_steps,
            eval_every,
        )

    if args.offline or num_rollouts > 0:
        t0 = time.perf_counter()
        for b in range(B):
            agent.save(agent_state, seed_dirs[b], b)
        log_time(loggers, "time/checkpoint_s", t0, size)

    for logger in loggers:
        logger.close()

    open(os.path.join(log_dir, "COMPLETE"), "w").close()


if __name__ == "__main__":
    main(tyro.cli(Args))
