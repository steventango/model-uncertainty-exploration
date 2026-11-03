import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import chex
import gymnasium
import jax.numpy as jnp
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import main
from offline_data import AREA_INDEX
from ppo import Transition

GOLDEN_DIR = Path("/tmp/model-uncertainty-exploration-golden")

TINY_PPO = [
    "--ppo.num-envs", "4",
    "--ppo.num-steps", "10",
    "--ppo.total-timesteps", "80",
    "--ppo.num-minibatches", "2",
    "--ppo.update-epochs", "1",
    "--ppo.hidden-dim", "16",
]  # fmt: skip
TINY_ENN = ["model:enn", "--model.update-steps", "10", "--model.hidden-dim", "16"]
TINY_BLR = ["model:blr", "--model.num-samples", "4"]

CASES = {
    "enn-pendulum": ["--env", "Pendulum-v1", "--max-steps", "400", *TINY_PPO, *TINY_ENN],
    "blr-cartpole-2seeds": [
        "--env", "CartPole-v1",
        "--num-seeds", "2",
        "--max-steps", "100",
        "--steps-per-rollout", "50",
        "--model-env-mode", "mean",
        "--explore-bonus", "std",
        "--alpha", "1.0",
        *TINY_PPO,
        *TINY_BLR,
    ],
    "enn-cartpole-terminating": [
        "--env", "CartPole-v1",
        "--max-steps", "100",
        "--steps-per-rollout", "50",
        *TINY_PPO,
        *TINY_ENN,
    ],
    "predict-reward-buffer-reset": [
        "--env", "Pendulum-v1",
        "--max-steps", "400",
        "--predict-reward-terminated",
        "--reset-source", "buffer",
        "--rollout-length", "20",
        "--uncertainty-threshold", "1.0",
        "--eval-every", "100",
        *TINY_PPO,
        *TINY_ENN,
    ],
    "model-free-cartpole": [
        "--env", "CartPole-v1",
        "--model-free",
        "--max-steps", "100",
        "--steps-per-rollout", "50",
        *TINY_PPO,
    ],
    "brax-inverted-pendulum": [
        "--env", "inverted_pendulum",
        "--max-steps", "100",
        "--steps-per-rollout", "50",
        *TINY_PPO,
        *TINY_ENN,
    ],
}  # fmt: skip


def _run(monkeypatch, log_dir, argv):
    monkeypatch.setattr(sys, "argv", ["main.py", "--log-dir", str(log_dir), *argv])
    main.main()


def _scalars(seed_dir, tag):
    acc = EventAccumulator(str(seed_dir))
    acc.Reload()
    return [e.value for e in acc.Scalars(tag)]


def _all_scalars(log_dir):
    """Every logged scalar except wall-clock timings, keyed by seed then tag."""
    out = {}
    for seed_dir in sorted(log_dir.glob("seed_*")):
        acc = EventAccumulator(str(seed_dir), size_guidance={"scalars": 0})
        acc.Reload()
        out[seed_dir.name] = {
            tag: [[e.step, e.value] for e in acc.Scalars(tag)]
            for tag in acc.Tags()["scalars"]
            if not tag.startswith("time/")
        }
    return out


def _check_golden(name, actual):
    """Compare ``{seed: {tag: [[step, value], ...]}}`` with GOLDEN_DIR/<name>.json."""
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        warnings.warn(f"recorded golden {path}")
        return
    expected = json.loads(path.read_text())
    actual, expected = (
        {
            seed: {tag: np.asarray(points) for tag, points in tags.items()}
            for seed, tags in d.items()
        }
        for d in (actual, expected)
    )
    chex.assert_trees_all_close(actual, expected)


def _check_outputs(log_dir, *, model, eval_return=True):
    assert (log_dir / "COMPLETE").exists()
    seed_dirs = sorted(log_dir.glob("seed_*"))
    assert seed_dirs
    for seed_dir in seed_dirs:
        ckpt = seed_dir / "checkpoint"
        assert (ckpt / "ppo_explore" / "network").is_dir()
        assert (ckpt / "ppo_eval" / "network").is_dir()
        assert (ckpt / "model").is_dir() == model
        if eval_return:
            returns = _scalars(seed_dir, "eval/mean_return")
            assert returns and all(math.isfinite(r) for r in returns)
    return seed_dirs


@pytest.mark.parametrize("case", CASES)
def test_online(monkeypatch, tmp_path, case):
    argv = CASES[case]
    _run(monkeypatch, tmp_path, argv)
    seed_dirs = _check_outputs(tmp_path, model="--model-free" not in argv)
    num_seeds = int(argv[argv.index("--num-seeds") + 1]) if "--num-seeds" in argv else 1
    assert len(seed_dirs) == num_seeds
    _check_golden(case, _all_scalars(tmp_path))


def _fake_plant_dataset(num_episodes=4, ep_len=14, act_dim=1, obs_dim=AREA_INDEX + 1):
    """Synthetic stand-in for load_offline_transitions (no minari dataset in CI)."""
    rng = np.random.default_rng(0)
    area = np.cumsum(rng.uniform(0.0, 0.1, (num_episodes, ep_len + 1)), axis=1)
    obs = area[:, :-1].reshape(-1, 1).astype(np.float32)
    next_obs = area[:, 1:].reshape(-1, 1).astype(np.float32)
    n = obs.shape[0]
    action = rng.uniform(0.4, 1.3, (n, act_dim)).astype(np.float32)
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros((num_episodes, ep_len), dtype=bool)
    truncated[:, -1] = True
    zeros = jnp.zeros(n)
    transitions = Transition(
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated.reshape(-1)),
        action=jnp.asarray(action),
        value=zeros,
        next_value=zeros,
        reward=jnp.asarray(next_obs[:, 0] - obs[:, 0]),
        log_prob=zeros,
        obs=jnp.asarray(obs),
        info={"next_obs": jnp.asarray(next_obs)},
        raw_obs=jnp.asarray(obs),
        raw_action=jnp.asarray(action),
        raw_reward=jnp.asarray(next_obs[:, 0] - obs[:, 0]),
    )
    action_space = gymnasium.spaces.Box(0.4, 1.3, (act_dim,), np.float32)
    observation_space = gymnasium.spaces.Box(0.0, 2.0, (obs_dim,), np.float32)
    return transitions, action_space, observation_space, ep_len


def test_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main, "load_offline_transitions", lambda _: _fake_plant_dataset()
    )
    _run(monkeypatch, tmp_path, ["--offline", *TINY_PPO, *TINY_ENN])
    _check_outputs(tmp_path, model=True, eval_return=False)
    _check_golden("offline", _all_scalars(tmp_path))


# classic_plan_every_tenth on Pendulum, cut from 100 plans to 10.
GPU_CASES = {
    "gpu-pendulum-every-tenth": [
        "--env", "Pendulum-v1",
        "--num-seeds", "2",
        "--steps-per-rollout", "20",
        "--max-steps", "200",
    ],
}  # fmt: skip
DETERMINISTIC_XLA = (
    "--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true"
)


@pytest.mark.gpu
@pytest.mark.parametrize("case", GPU_CASES)
def test_gpu(tmp_path, case):
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} {DETERMINISTIC_XLA}".strip()
    cmd = [sys.executable, "main.py", "--log-dir", str(tmp_path), *GPU_CASES[case]]
    subprocess.run(cmd, env=env, check=True)
    _check_outputs(tmp_path, model=True)
    _check_golden(case, _all_scalars(tmp_path))
