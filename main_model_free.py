"""Vanilla (model-free) PPO baseline on classic-control envs.

Trains PPO directly against the real environment (no world model, no Dyna
loop) so its ``eval/mean_return`` learning curve can be overlaid against the
model-based runs from ``main.py`` in ``scripts/plot_learning_curves.py`` to
illustrate M-PPO's sample-efficiency advantage.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import tyro
from flax import nnx
from gymnax.environments import spaces

import evaluation
from config import PPOConfig, ppo_config_dict
from logger import ExperimentLogger, log_eval
from ppo import make_rollout, make_train, make_train_state
from wrappers import ClipAction, LogWrapper, VecEnv


def _wrap_env(env, discrete):
    wrapped = LogWrapper(env)
    if not discrete:
        wrapped = ClipAction(wrapped)
    return VecEnv(wrapped)


@partial(jax.jit, static_argnums=1)
def vsplit(keys, num=2):
    return jax.vmap(lambda k: jax.random.split(k, num), out_axes=1)(keys)


@dataclass(frozen=True)
class Args:
    env: str = "Pendulum-v1"
    seed: int = 0
    num_seeds: int = 1
    total_timesteps: float = 100_000
    num_evals: int = 20
    log_dir: str | None = None
    label: str = "model_free"
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(num_envs=1, num_steps=128)
    )


def main():
    args = tyro.cli(Args)

    B = args.num_seeds
    seeds = args.seed + jnp.arange(B)
    seed_keys = jax.vmap(jax.random.key)(seeds)

    base_env, env_params = gymnax.make(args.env)
    action_space = base_env.action_space(env_params)
    discrete = isinstance(action_space, spaces.Discrete)
    env = _wrap_env(base_env, discrete)

    config = ppo_config_dict(
        args.ppo, env_name=args.env, seed=args.seed, offline=False
    )

    num_evals = max(1, args.num_evals)
    chunk_size = max(
        config["NUM_ENVS"] * config["NUM_STEPS"],
        int(args.total_timesteps) // num_evals,
    )
    num_chunks = int(args.total_timesteps) // chunk_size
    # NUM_UPDATES/MINIBATCH_SIZE are derived once, at make_train() call time,
    # from TOTAL_TIMESTEPS — so this must be the per-chunk budget, not the
    # overall total, since train_jit below is built once and reused per chunk.
    config["TOTAL_TIMESTEPS"] = chunk_size

    eval_config = {
        **config,
        "NUM_ENVS": 100,
        "NUM_STEPS": env_params.max_steps_in_episode,
    }
    batched_eval = evaluation.make_batched_evaluate_policy(
        eval_config, env, env_params, make_rollout(eval_config, env, env_params, training=False)
    )

    seed_keys, model_keys = vsplit(seed_keys)
    train_state = nnx.vmap(
        lambda key: make_train_state(config, env, env_params, nnx.Rngs(params=key))
    )(model_keys)

    train_jit = nnx.jit(nnx.vmap(make_train(env, config), in_axes=(0, None, 0)))

    dt = datetime.now()
    log_dir = (
        args.log_dir or f"runs/{dt.strftime('%Y%m%d')}/{args.env}/{dt.strftime('%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"env": args.env, "model": "model_free", "label": args.label}, f)

    seed_dirs, loggers = [], []
    for b in range(B):
        seed_dir = f"{log_dir}/seed_{int(seeds[b])}"
        logger = ExperimentLogger(seed_dir)
        logger.log_hparams(
            args={"env": args.env, "seed": int(seeds[b]), "label": args.label},
            ppo=config,
            run={"num_chunks": num_chunks, "chunk_size": chunk_size},
        )
        seed_dirs.append(seed_dir)
        loggers.append(logger)

    data_count = 0
    seed_keys, eval_keys = vsplit(seed_keys)
    log_eval(loggers, batched_eval, train_state, eval_keys, data_count)

    for j in range(num_chunks):
        seed_keys, train_seed = vsplit(seed_keys)
        out = train_jit(train_state, env_params, train_seed)
        jax.block_until_ready(out)
        train_state = out["runner_state"][0]
        data_count += chunk_size

        for b in range(B):
            loggers[b].log_ppo_returns(
                _seed_slice_metrics(out["metrics"], b),
                "ppo/train_return",
                j,
                config["NUM_ENVS"],
                config["NUM_STEPS"],
                chunk_size,
            )

        seed_keys, eval_keys = vsplit(seed_keys)
        log_eval(loggers, batched_eval, train_state, eval_keys, data_count)
        print(f"[{args.env}] {data_count}/{int(args.total_timesteps)} real steps")

    for logger in loggers:
        logger.close()
    open(os.path.join(log_dir, "COMPLETE"), "w").close()


def _seed_slice_metrics(metrics, b):
    return jax.tree_util.tree_map(lambda x: x[b], metrics)


if __name__ == "__main__":
    main()
