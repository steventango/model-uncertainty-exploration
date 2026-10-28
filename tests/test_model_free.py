"""Model-free online PPO (ppo.make_online_train, main.py --model-free)."""

import os
import subprocess
import sys

import gymnax
import jax
import numpy as np
import pytest
from flax import nnx

import ppo
from config import PPOConfig, ppo_config_dict
from wrappers import LogWrapper, VecEnv

# This process already holds most of the GPU; let main.py allocate on demand.
_ENV = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def _setup(B, num_steps, num_minibatches):
    config = ppo_config_dict(
        PPOConfig(num_minibatches=num_minibatches),
        env_name="MountainCar-v0",
        seed=0,
        offline=False,
    )
    config = {
        **config,
        "NUM_ENVS": 1,
        "NUM_STEPS": num_steps,
        "MINIBATCH_SIZE": num_steps // num_minibatches,
    }
    env, env_params = gymnax.make("MountainCar-v0")
    env = VecEnv(LogWrapper(env))

    @nnx.vmap
    def build(key):
        return ppo.make_train_state(config, env, env_params, nnx.Rngs(params=key))

    train_state = build(jax.random.split(jax.random.key(0), B))
    reset_keys = jax.random.split(jax.random.key(1), (B, 1))
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
    rollout = nnx.jit(
        nnx.vmap(ppo.make_rollout(config, env, env_params, training=False))
    )
    runner_state = (
        train_state,
        env_state,
        obsv,
        jax.random.split(jax.random.key(2), B),
    )
    return config, train_state, rollout, runner_state


def test_online_train_updates_batched_state_in_place():
    """main._run_model_free relies on nnx.vmap writing updates back (warm start)."""
    B, num_steps, num_minibatches = 2, 40, 4
    config, train_state, rollout, runner_state = _setup(B, num_steps, num_minibatches)
    _, traj_batch = rollout(runner_state)
    before = jax.tree.map(np.array, nnx.state(train_state[0], nnx.Param))

    train = nnx.jit(nnx.vmap(ppo.make_online_train(config)))
    _, losses = train(train_state, traj_batch, jax.random.split(jax.random.key(3), B))

    assert set(losses) == {"total", "value", "actor", "entropy"}
    for v in losses.values():
        assert v.shape == (B, config["UPDATE_EPOCHS"] * num_minibatches)
        assert np.isfinite(np.asarray(v)).all()
    after = nnx.state(train_state[0], nnx.Param)
    changed = jax.tree.map(lambda a, b: np.any(a != b), before, after)
    assert all(jax.tree.leaves(changed))


def test_main_rejects_indivisible_minibatches(tmp_path):
    argv = [
        sys.executable,
        "main.py",
        "--env",
        "MountainCar-v0",
        "--model-free",
        "--ppo.num-minibatches",
        "32",
        "--log-dir",
        str(tmp_path),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, env=_ENV)
    assert result.returncode != 0
    assert "must divide" in result.stderr


@pytest.mark.slow
def test_main_smoke(tmp_path):
    argv = [
        sys.executable,
        "main.py",
        "--env",
        "MountainCar-v0",
        "--model-free",
        "--ppo.num-minibatches",
        "4",
        "--max-steps",
        "400",
        "--num-seeds",
        "2",
        "--log-dir",
        str(tmp_path),
    ]
    subprocess.run(argv, check=True, env=_ENV)
    assert (tmp_path / "COMPLETE").is_file()
    assert (tmp_path / "seed_0" / "checkpoint" / "ppo_eval").is_dir()
