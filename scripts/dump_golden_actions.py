"""Dump golden actions for a P1 deployment parity test (see EXPERIMENTS).

For each checkpoint (per reward mode) and PPO head (explore / eval) this
restores the network + obs normalizer exactly as saved by main.py and records,
over a grid of observations spanning that dataset's log-area range:
  - mean_actions: clip(pi.mean(), action_min, action_max)  (exploit-zone path)
  - std: exp(log_std)
  - sampled_actions: clip(pi.sample(seed=fold_in(PRNGKey(seed), day_ordinal)))
    (explore-zone path; the key derivation must match plant-rl's PPOPolicy)

The output JSON is copied into plant-rl's tests/test_data/<experiment>P1/ so the
deployed agents can be asserted against these references.

Usage (from the repo root):
    uv run python scripts/dump_golden_actions.py --experiment E21 --out golden_actions.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networks import ActorCritic  # noqa: E402
from normalization import NormalizeVecObs  # noqa: E402

# Deployment clip bounds (E19/E20/E21 ops convention) — keep equal to the
# action_min/action_max in plant-rl's experiments/online/E*/P1/*.json.
ACTION_MIN, ACTION_MAX = 0.4, 1.3
NUM_GRID = 9
DAYS = [0.0, 7.0, 14.0]  # day-index values for 2-D (masked) observations
SEED = 0
DAY_ORDINALS = [739078, 739079, 739092]  # fixed dates for sampled-action parity

# Per-experiment presets: the deployed checkpoints, their policy obs dims, and
# the log-area range of the dataset each was trained on (the grid must span the
# range the deployed policy will actually see).
EXPERIMENTS = {
    "E20": {
        "area_range": (-2.6221607, 2.4120054),  # plant-data/visu-v2-v27
        "checkpoints": {
            "analytic": "runs/20260710/plant_plant-data_visu-v2-v27/gamma0.8_enn_ln/seed_0/checkpoint",
            "masked": "runs/20260711/plant_plant-data_visu-v2-v27/gamma0.8_enn_ln_masked_timeobs_nextarea/seed_0/checkpoint",
        },
        "obs_dims": {"analytic": 1, "masked": 2},
    },
    "E21": {
        "area_range": (-3.0966384410858154, 2.4120266437530518),  # plant-data/visu-v28
        "checkpoints": {
            "analytic": "runs/20260807/plant_plant-data_visu-v28/gamma0.8_enn_ln/seed_0/checkpoint",
            "masked_log": "runs/20260731/plant_plant-data_visu-v28/gamma0.8_enn_ln_masked_log/seed_0/checkpoint",
        },
        "obs_dims": {"analytic": 1, "masked_log": 2},
    },
    "E22": {
        # e18e21-daily-v29 / e18e21-daily-diff-v29 share a log-area range; the
        # all-daily grid is wider but the shared grid below spans what all three
        # policies will actually see.
        "area_range": (-2.6446533203125, 2.7040624618530273),
        "checkpoints": {
            "standard": "runs/20260827/plant_plant-data_e18e21-daily-v29/gamma0.8_enn_ln_diffsetup_analytic/seed_0/checkpoint",
            "differential": "runs/20260827/plant_plant-data_e18e21-daily-diff-v29/gamma0.8_enn_ln_diffsetup_analytic_diff/seed_0/checkpoint",
            "masked_e21": "runs/20260827/plant_plant-data_e18e21-daily-v29/gamma0.8_enn_ln_diffsetup_masked_log_e21/seed_0/checkpoint",
            "all_data": "runs/20260827/plant_plant-data_all-daily-v29/gamma0.8_enn_ln_diffsetup_analytic/seed_0/checkpoint",
        },
        # All three carry the day: the absolute arms were trained with
        # --include-time so the differential comparison isn't confounded by a
        # different state space.
        "obs_dims": {
            "standard": 2,
            "differential": 2,
            "masked_e21": 2,
            "all_data": 2,
        },
    },
    "E23": {
        # E22's recipe retrained on e18e21e22-daily{,-diff}-v30 (same parquet,
        # same obs+next_obs log-area range). No all_data arm: E23 Z5 is a
        # constant-spectrum zone.
        "area_range": (-2.817924976348877, 2.737022876739502),
        "checkpoints": {
            "standard": "runs/20260928/plant_plant-data_e18e21e22-daily-v30/gamma0.8_enn_ln_diffsetup_analytic/seed_0/checkpoint",
            "differential": "runs/20260928/plant_plant-data_e18e21e22-daily-diff-v30/gamma0.8_enn_ln_diffsetup_analytic_diff/seed_0/checkpoint",
            "masked_e21": "runs/20260928/plant_plant-data_e18e21e22-daily-v30/gamma0.8_enn_ln_diffsetup_masked_log_e21/seed_0/checkpoint",
        },
        "obs_dims": {"standard": 2, "differential": 2, "masked_e21": 2},
    },
}


def restore_head(ckpt_dir: str, head: str, obs_dim: int):
    checkpointer = ocp.StandardCheckpointer()
    network = ActorCritic(
        obs_dim, 1, 64, activation="tanh", use_layer_norm=False, rngs=nnx.Rngs(0)
    )
    obs_norm = NormalizeVecObs(jnp.zeros(obs_dim))
    net_gd, net_state = nnx.split(network)
    on_gd, on_state = nnx.split(obs_norm)
    net_state = checkpointer.restore(
        os.path.join(ckpt_dir, head, "network"), target=net_state
    )
    on_state = checkpointer.restore(
        os.path.join(ckpt_dir, head, "obs_norm"), target=on_state
    )
    return nnx.merge(net_gd, net_state), nnx.merge(on_gd, on_state)


def obs_grid(obs_dim: int, area_range: tuple[float, float]) -> np.ndarray:
    areas = np.linspace(*area_range, NUM_GRID, dtype=np.float32)
    if obs_dim == 1:
        return areas[:, None]
    return np.asarray([[a, d] for d in DAYS for a in areas], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--experiment",
        default="E21",
        choices=sorted(EXPERIMENTS),
        help="Which deployment's checkpoints/dataset range to dump.",
    )
    args = parser.parse_args()

    preset = EXPERIMENTS[args.experiment]
    checkpoints = preset["checkpoints"]
    obs_dims = preset["obs_dims"]
    area_range = preset["area_range"]

    golden: dict = {
        "experiment": args.experiment,
        "action_min": ACTION_MIN,
        "action_max": ACTION_MAX,
        "seed": SEED,
        "day_ordinals": DAY_ORDINALS,
        "area_range": list(area_range),
        "checkpoints": {k: os.path.abspath(v) for k, v in checkpoints.items()},
    }

    for mode, ckpt in checkpoints.items():
        ckpt = os.path.abspath(ckpt)
        obs_dim = obs_dims[mode]
        grid = obs_grid(obs_dim, area_range)
        golden[mode] = {"obs_dim": obs_dim, "grid": grid.tolist()}
        for head in ("ppo_explore", "ppo_eval"):
            network, obs_norm = restore_head(ckpt, head, obs_dim)
            std = float(np.exp(np.asarray(network.actor.log_std.get_value()))[0])

            mean_actions, sampled = [], []
            for obs in grid:
                # Mirror plant-rl PPOPolicy._select_action: single-obs forward.
                pi, _ = network(obs_norm.normalize(jnp.asarray(obs)))
                mean_actions.append(
                    float(np.clip(np.asarray(pi.mean())[0], ACTION_MIN, ACTION_MAX))
                )
                per_date = []
                for ordinal in DAY_ORDINALS:
                    key = jax.random.fold_in(jax.random.PRNGKey(SEED), ordinal)
                    a = np.asarray(pi.sample(seed=key))[0]
                    per_date.append(float(np.clip(a, ACTION_MIN, ACTION_MAX)))
                sampled.append(per_date)

            golden[mode][head] = {
                "std": std,
                "mean_actions": mean_actions,
                "sampled_actions": sampled,
            }
            print(
                f"{mode}/{head}: std={std:.5f} "
                f"mean range [{min(mean_actions):.4f}, {max(mean_actions):.4f}]"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
