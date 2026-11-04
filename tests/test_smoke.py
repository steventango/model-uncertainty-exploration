import dataclasses
import json
import math
import warnings
from pathlib import Path

import chex
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import main
from config import Args, BLRConfig, ENNConfig, PPOConfig, RNDConfig
from offline_data import AREA_INDEX
from ppo import Transition

GOLDEN_DIR = Path("/tmp/model-uncertainty-exploration-golden")

TINY_PPO = PPOConfig(
    num_envs=4,
    num_steps=10,
    total_timesteps=80,
    num_minibatches=2,
    update_epochs=1,
    hidden_dim=16,
)
TINY_ENN = ENNConfig(update_steps=10, hidden_dim=16)
TINY_BLR = BLRConfig(num_samples=4)

CASES = {
    "enn-pendulum": Args(
        cpu=True, env="Pendulum-v1", max_steps=400, ppo=TINY_PPO, model=TINY_ENN
    ),
    "blr-cartpole-2seeds": Args(
        cpu=True,
        env="CartPole-v1",
        num_seeds=2,
        max_steps=100,
        steps_per_rollout=50,
        model_env_mode="mean",
        explore_bonus="std",
        alpha=1.0,
        ppo=TINY_PPO,
        model=TINY_BLR,
    ),
    "enn-cartpole-terminating": Args(
        cpu=True,
        env="CartPole-v1",
        max_steps=100,
        steps_per_rollout=50,
        ppo=TINY_PPO,
        model=TINY_ENN,
    ),
    "predict-reward-buffer-reset": Args(
        cpu=True,
        env="Pendulum-v1",
        max_steps=400,
        predict_reward_terminated=True,
        reset_source="buffer",
        rollout_length=20,
        uncertainty_threshold=1.0,
        eval_every=100,
        ppo=TINY_PPO,
        model=TINY_ENN,
    ),
    "rnd-pendulum-2seeds": Args(
        cpu=True,
        env="Pendulum-v1",
        num_seeds=2,
        max_steps=400,
        explore_bonus="rnd",
        ppo=TINY_PPO,
        rnd=RNDConfig(update_steps=10),
        model=TINY_ENN,
    ),
    "model-free-cartpole": Args(
        cpu=True,
        env="CartPole-v1",
        model_free=True,
        max_steps=100,
        steps_per_rollout=50,
        ppo=TINY_PPO,
    ),
    "rnd-model-free-cartpole": Args(
        cpu=True,
        env="CartPole-v1",
        model_free=True,
        explore_bonus="rnd",
        max_steps=100,
        steps_per_rollout=50,
        ppo=TINY_PPO,
        rnd=RNDConfig(update_steps=10),
    ),
    "brax-inverted-pendulum": Args(
        cpu=True,
        env="inverted_pendulum",
        max_steps=100,
        steps_per_rollout=50,
        ppo=TINY_PPO,
        model=TINY_ENN,
    ),
    "pendulum-every-tenth": Args(
        env="Pendulum-v1", num_seeds=2, steps_per_rollout=20, max_steps=200
    ),
}


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


def _check_outputs(log_dir, *, model):
    scalars = _all_scalars(log_dir)
    assert scalars
    for seed, tags in scalars.items():
        ckpt = log_dir / seed / "checkpoint"
        assert (ckpt / "ppo_explore" / "network").is_dir()
        assert (ckpt / "ppo_eval" / "network").is_dir()
        assert (ckpt / "model").is_dir() == model
        assert all(math.isfinite(v) for points in tags.values() for _, v in points)
    return scalars


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(name, marks=() if args.cpu else pytest.mark.gpu)
        for name, args in CASES.items()
    ],
)
def test_online(tmp_path, case):
    args = CASES[case]
    main.main(dataclasses.replace(args, log_dir=str(tmp_path)))
    assert jax.default_backend() == ("cpu" if args.cpu else "gpu")
    scalars = _check_outputs(tmp_path, model=not args.model_free)
    assert len(scalars) == args.num_seeds
    assert all("eval/mean_return" in tags for tags in scalars.values())
    _check_golden(case, scalars)


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
    main.main(
        Args(
            cpu=True, offline=True, log_dir=str(tmp_path), ppo=TINY_PPO, model=TINY_ENN
        )
    )
    _check_golden("offline", _check_outputs(tmp_path, model=True))
