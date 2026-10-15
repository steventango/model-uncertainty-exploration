"""Sanity-check the learned reward head against the dataset's reward column.

``--reward-mode learned`` puts the reward on the dynamics model's reward head
(WorldModel.build_targets), so a policy comparison across reward definitions is
only meaningful if each head actually fits its own dataset's reward. This
restores a run's saved model and reports the fit on the training transitions.

Usage:
    uv run python scripts/check_learned_reward_fit.py \
        --checkpoint runs/20260826/plant_plant-data_e18e21-daily-diff-v29/gamma0.8_enn_ln_learned/seed_0/checkpoint \
        --dataset plant-data/e18e21-daily-diff-v29
"""

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.enn import ENNModel  # noqa: E402
from networks import ENN  # noqa: E402
from offline_data import load_offline_transitions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="<run>/seed_N/checkpoint")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--layer-norm", action="store_true", default=True)
    args = ap.parse_args()

    tr, action_space, _, _ = load_offline_transitions(args.dataset, include_time=True)
    obs_dim = tr.obs.shape[-1]
    action_dim = action_space.shape[0]

    # Same architecture main.py built (models.enn._make_batched_model), unbatched:
    # the checkpoint is one seed's slice of the vmapped model.
    enn = ENN(
        obs_dim + action_dim,
        args.hidden_dim,
        15,
        5,
        obs_dim + 2,
        8,
        use_layer_norm=args.layer_norm,
        rngs=nnx.Rngs(params=jax.random.key(0)),
    )
    model = ENNModel(
        enn,
        obs_dim + action_dim,
        obs_dim,
        act_dim=None,
        predict_reward_terminated=True,
    )
    gd, state = nnx.split(model)
    state = ocp.StandardCheckpointer().restore(
        os.path.abspath(os.path.join(args.checkpoint, "model")), target=state
    )
    model = nnx.merge(gd, state)

    x = model.normalize_input(model.build_input(tr.obs, tr.action))
    y = jax.vmap(model.predict_mean)(x)
    pred = np.asarray(model.denormalize_reward(y[..., -2]))
    true = np.asarray(tr.reward)
    resid = true - pred
    print(f"dataset {args.dataset}  N={len(true)}")
    print(f"  reward  mean {true.mean():+.5f}  std {true.std():.5f}")
    print(f"  pred    mean {pred.mean():+.5f}  std {pred.std():.5f}")
    print(
        f"  resid   bias {resid.mean():+.5f}  std {resid.std():.5f}  "
        f"R2 {1 - resid.var() / true.var():+.4f}  corr {np.corrcoef(true, pred)[0, 1]:.4f}"
    )
    # Day trend: the differential reward subtracts a per-(experiment, day)
    # control, so its day profile should be flat where the absolute reward's is not.
    day = np.asarray(tr.obs[:, 1]).astype(int)
    print("  reward by day (true / predicted):")
    for d in range(15):
        m = day == d
        if m.sum() == 0:
            continue
        print(
            f"    day {d:2d}  n={m.sum():5d}  true {true[m].mean():+.5f}  "
            f"pred {pred[m].mean():+.5f}"
        )


if __name__ == "__main__":
    main()
