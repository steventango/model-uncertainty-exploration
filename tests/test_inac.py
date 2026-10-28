"""InAC port (inac.py): update, offline/online training loops, main.py wiring."""

import os
import subprocess
import sys

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import inac
from config import InACConfig, inac_config_dict
from wrappers import LogWrapper, VecEnv

# This process already holds most of the GPU; let main.py allocate on demand.
_ENV = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def _config(**overrides):
    c = InACConfig(hidden_dim=32, batch_size=64, **overrides)
    return inac_config_dict(c, env_name="MountainCar-v0", seed=0, offline=False)


def _agent(obs_dim, action_dim, discrete, config, seed=0):
    rngs = nnx.Rngs(seed)
    agent = inac.InACAgent(
        obs_dim, action_dim, config["HIDDEN_DIM"], discrete, rngs=rngs
    )
    return agent, inac.InACOptimizers(agent, config["LR"])


def _snapshot(module):
    """Copy of a module's params (nnx.state aliases the live Variables)."""
    return jax.tree.map(np.array, nnx.state(module, nnx.Param))


def _random_batch(key, n, obs_dim, action_dim, discrete):
    k = jax.random.split(key, 4)
    obs = jax.random.normal(k[0], (n, obs_dim))
    if discrete:
        action = jax.random.randint(k[1], (n,), 0, action_dim)
    else:
        action = jax.random.uniform(k[1], (n, action_dim), minval=-1, maxval=1)
    reward = jax.random.normal(k[2], (n,))
    terminated = jax.random.bernoulli(k[3], 0.1, (n,)).astype(jnp.float32)
    return obs, action, reward, terminated, obs + 0.1


@pytest.mark.parametrize("discrete", [True, False])
def test_update_finite(discrete):
    config = _config()
    agent, opts = _agent(2, 3, discrete, config)
    batch = _random_batch(jax.random.key(0), 64, 2, 3, discrete)
    q_target_before = _snapshot(agent.q_target)
    losses = nnx.jit(lambda a, o, b, k: inac.inac_update(a, o, b, k, config))(
        agent, opts, batch, jax.random.key(1)
    )
    assert set(losses) == {
        "beta",
        "actor",
        "critic",
        "value",
        "q_info",
        "v_info",
        "logp_info",
    }
    for v in losses.values():
        assert v.shape == () and jnp.isfinite(v)
    # Polyak moved the target toward the (updated) online critic.
    moved = jax.tree.map(
        lambda a, b: jnp.any(a != b),
        q_target_before,
        nnx.state(agent.q_target, nnx.Param),
    )
    assert any(jax.tree.leaves(moved))


def test_offline_bandit_picks_best_in_sample_action():
    """One-step bandit, uniform behaviour data: action 2 pays 1, others 0."""
    config = _config(offline_updates=1500, tau=0.1, lr=1e-3)
    n = 3000
    k_obs, k_act = jax.random.split(jax.random.key(0))
    obs = jax.random.normal(k_obs, (n, 2))
    action = jax.random.randint(k_act, (n, 1), 0, 3)  # collate_rollout layout
    reward = (action[:, 0] == 2).astype(jnp.float32)
    buffer = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "terminated": jnp.ones(n, dtype=bool),
        "next_obs": obs,
    }
    agent, opts = _agent(2, 3, True, config)
    train = nnx.jit(inac.make_offline_train(config))
    train((agent, opts, None, None), buffer, jnp.int32(n), jax.random.key(1))
    probs = agent.pi(obs[:200]).probs
    assert float(probs[:, 2].mean()) > 0.9


def test_offline_train_updates_batched_state_in_place():
    """main._run_model_free relies on nnx.vmap writing updates back (warm start)."""
    config = _config(offline_updates=5)
    env, env_params = gymnax.make("MountainCar-v0")
    env = VecEnv(LogWrapper(env))

    @nnx.vmap
    def build(key):
        return inac.make_train_state(config, env, env_params, nnx.Rngs(params=key))

    B, n = 2, 128
    train_state = build(jax.random.split(jax.random.key(0), B))
    before = _snapshot(train_state[0].pi)
    buffer = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (B,) + x.shape),
        dict(
            zip(
                ("obs", "action", "reward", "terminated", "next_obs"),
                _random_batch(jax.random.key(1), n, 2, 3, True),
            )
        ),
    )
    train = nnx.jit(nnx.vmap(inac.make_offline_train(config), in_axes=(0, 0, None, 0)))
    _, losses = train(
        train_state, buffer, jnp.int32(n), jax.random.split(jax.random.key(2), B)
    )
    assert losses["actor"].shape == (B, 5)
    after = nnx.state(train_state[0].pi, nnx.Param)
    changed = jax.tree.map(lambda a, b: jnp.any(a != b), before, after)
    assert all(jax.tree.leaves(changed))


def test_online_train_metric_layout():
    config = _config(num_envs=4, num_iters=20, learning_starts=32, updates_per_iter=2)
    env, env_params = gymnax.make("MountainCar-v0")
    env = VecEnv(LogWrapper(env))
    train_state = inac.make_train_state(config, env, env_params, nnx.Rngs(0))
    out = nnx.jit(lambda ts, k: inac.make_train(env, config)(ts, env_params, k))(
        train_state, jax.random.key(0)
    )
    assert out["metrics"]["returned_episode"].shape == (20, 4)
    assert out["losses"]["critic"].shape == (20 - 32 // 4,)
    assert np.isfinite(np.asarray(out["losses"]["critic"])).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "extra",
    [
        ["--model-free", "--inac.offline-updates", "20"],
        [
            "--inac.num-iters",
            "40",
            "--inac.learning-starts",
            "64",
            "model:enn",
            "--model.update-steps",
            "20",
        ],
    ],
    ids=["model_free", "model_env"],
)
def test_main_smoke(tmp_path, extra):
    argv = [
        sys.executable,
        "main.py",
        "--env",
        "MountainCar-v0",
        "--solver",
        "inac",
        "--max-steps",
        "400",
        "--num-seeds",
        "2",
        "--log-dir",
        str(tmp_path),
        *extra,
    ]
    subprocess.run(argv, check=True, env=_ENV)
    assert (tmp_path / "COMPLETE").is_file()
