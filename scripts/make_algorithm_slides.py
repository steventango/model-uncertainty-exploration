#!/usr/bin/env python3
"""Presentation assets for the algorithm figure (slides A and B).

Slide A: one posterior sample = one self-consistent imagined plant (a single
         14-day growth curve).
Slide B: 2048 of them, with one shared policy acting in every imagined plant.

Both are *real* rollouts in the deployed E21/P1 masked_log world model: the ENN
dynamics model and the eval policy are restored from the deployment checkpoint,
and each episode holds one fixed epistemic index z for all 14 days (exactly what
ModelEnvironment.reset_env does), so a curve is one coherent plant rather than
per-step noise.

Every rollout starts from the *same* day-0 area on purpose: the spread you see is
then model disagreement alone, not initial-condition scatter.

Usage:
    uv run python scripts/make_algorithm_slides.py --out-dir figures/slides
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from flax import nnx  # noqa: E402

from models.enn import ENNModel  # noqa: E402
from networks import ENN, ActorCritic  # noqa: E402
from normalization import NormalizeVecObs  # noqa: E402
from plant_env import PlantEnv, PlantEnvParams, _BASELINE_AREA_E21  # noqa: E402

CKPT = "/home/steven/Github/plant-rl/checkpoints/E21/P1/masked_log/checkpoint"

# Architecture read off the checkpoint's own array shapes (not guessed):
#   base.linear1 (3,64) + ln1  -> HIDDEN_DIM=64, USE_LAYER_NORM=True
#   base.linear2 (64,2)        -> out_features=2
#   epinet.learnable (75,15)   -> 3 + 64 input feats + INDEX_DIM=8, hidden 15
#   epinet.prior (8,3,5)       -> INDEX_DIM=8, PRIOR_HIDDEN_DIM=5
IN_FEATURES, OBS_DIM, OUT_FEATURES = 3, 2, 2
HIDDEN_DIM, LEARNABLE_HIDDEN, PRIOR_HIDDEN, INDEX_DIM = 64, 15, 5, 8
DYN_DIM = 2  # legacy width: model predicts the whole obs delta, env overwrites the day

# E21/P1 deployment constants (see scripts/dump_golden_actions.py).
AREA_MIN, AREA_MAX = -3.0966384410858154, 2.4120266437530518
ACTION_MIN, ACTION_MAX = 0.4, 1.3
HORIZON = 14


def restore(module, path):
    gd, state = nnx.split(module)
    state = ocp.StandardCheckpointer().restore(os.path.abspath(path), target=state)
    return nnx.merge(gd, state)


def load_model():
    enn = ENN(
        IN_FEATURES, HIDDEN_DIM, LEARNABLE_HIDDEN, PRIOR_HIDDEN,
        OUT_FEATURES, INDEX_DIM, use_layer_norm=True, rngs=nnx.Rngs(params=0),
    )
    model = ENNModel(
        enn, IN_FEATURES, OBS_DIM, act_dim=1,
        predict_reward_terminated=False, dyn_dim=DYN_DIM,
    )
    return restore(model, os.path.join(CKPT, "model"))


def load_policy(head="ppo_eval"):
    net = ActorCritic(OBS_DIM, 1, 64, activation="tanh", use_layer_norm=False,
                      rngs=nnx.Rngs(0))
    obs_norm = NormalizeVecObs(jnp.zeros(OBS_DIM))
    net = restore(net, os.path.join(CKPT, head, "network"))
    obs_norm = restore(obs_norm, os.path.join(CKPT, head, "obs_norm"))
    return net, obs_norm


def rollout(model, net, obs_norm, env, params, z, obs0):
    """One imagined plant: fixed epistemic index z held for all HORIZON days."""
    low, high = env.observation_space(params).low, env.observation_space(params).high

    def step(obs, _):
        pi, _ = net(obs_norm.normalize(obs[None]))
        action = jnp.clip(pi.mean()[0], ACTION_MIN, ACTION_MAX)
        x = model.single_input(obs, action)
        y = model.predict_sample(x, z)
        dyn = model.denormalize_delta_obs(y[: model.dyn_dim])
        nxt = jnp.clip(env.compose_next_obs(obs, dyn), low, high)
        return nxt, (obs[0], action[0])

    _, (areas, actions) = jax.lax.scan(step, obs0, None, length=HORIZON)
    return areas, actions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="figures/slides")
    ap.add_argument("--num-models", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_model()
    net, obs_norm = load_policy()
    env = PlantEnv(act_dim=1, reward_mode="masked_log", include_time=True, dyn_dim=DYN_DIM)
    params = PlantEnvParams(
        area_min=AREA_MIN, area_max=AREA_MAX,
        act_low=jnp.asarray([ACTION_MIN]), act_high=jnp.asarray([ACTION_MAX]),
        max_steps_in_episode=HORIZON,
    )

    obs0 = jnp.asarray([float(jnp.log(_BASELINE_AREA_E21[0])), 0.0], dtype=jnp.float32)
    z = jax.random.normal(jax.random.PRNGKey(args.seed), (args.num_models, INDEX_DIM))

    areas, actions = jax.vmap(rollout, in_axes=(None, None, None, None, None, 0, None))(
        model, net, obs_norm, env, params, z, obs0
    )
    areas = np.asarray(areas)      # (M, T)
    actions = np.asarray(actions)  # (M, T)

    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "rollouts.npz")
    np.savez(npz, areas=areas, actions=actions, obs0=np.asarray(obs0))
    print(f"saved {npz}  areas={areas.shape} actions={actions.shape}")
    print(f"day-0 log area {obs0[0]:.3f}; day-13 spread "
          f"[{areas[:, -1].min():.3f}, {areas[:, -1].max():.3f}] "
          f"std={areas[:, -1].std():.3f}")


if __name__ == "__main__":
    main()
