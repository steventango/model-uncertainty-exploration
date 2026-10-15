"""Compare the plant policies trained on the three plant-data v29 datasets.

The point of the v29 batch is the differential reward

    R_t = (Δlog A_pol − Δlog A_ctl) − λ·(E_t − E_ctl)/E_ctl

with the control resolved per (experiment, day) — E18 rows against E18 Z11, E21
rows against E21 Z11. ``e18e21-daily-v29`` and ``e18e21-daily-diff-v29`` are
built from the SAME parquet and differ only in which reward column was exported,
so any difference between their policies is attributable to the reward
definition alone.

This dumps each run's ppo_eval mean action over a (log-area × day) grid, per
seed, and reports the between-dataset gap against the between-seed spread — the
latter is the noise floor the gap has to clear to mean anything.

Usage:
    uv run python scripts/compare_v29_reward_policies.py \
        --run-root runs/20260826 --label gamma0.8_enn_ln_learned
"""

import argparse
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offline_data import load_offline_transitions  # noqa: E402
from scripts.dump_golden_actions import ACTION_MIN, ACTION_MAX, restore_head  # noqa: E402

DATASETS = ("all-daily-v29", "e18e21-daily-v29", "e18e21-daily-diff-v29")
NUM_AREA = 25
DAYS = (0, 3, 7, 11, 14)


def area_range(dataset: str) -> tuple[float, float]:
    """log-area range of the transitions actually trained on.

    Taken from the data, not from the Minari declared space, for the same reason
    main.py does (plant-data derives the declared bounds from pre-filter stats).
    """
    tr = load_offline_transitions(f"plant-data/{dataset}", include_time=True)[0]
    areas = np.concatenate(
        [np.asarray(tr.obs)[:, 0], np.asarray(tr.info["next_obs"])[:, 0]]
    )
    return float(areas.min()), float(areas.max())


def mean_actions(ckpt: str, head: str, grid: np.ndarray) -> np.ndarray:
    """Raw pi.mean() per grid point — NOT clipped.

    The deployment clip to [ACTION_MIN, ACTION_MAX] hides how far outside the
    operating range a policy wants to go, which is exactly what a reward change
    can move. Clipping is reported separately by ``clip_fraction``.
    """
    network, obs_norm = restore_head(ckpt, head, grid.shape[-1])
    out = []
    for obs in grid:
        pi, _ = network(obs_norm.normalize(jnp.asarray(obs)))
        out.append(float(np.asarray(pi.mean())[0]))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="runs/20260827")
    ap.add_argument("--label", default="gamma0.8_enn_ln_diffsetup")
    ap.add_argument(
        "--mode",
        nargs=3,
        default=["analytic", "analytic", "analytic_diff"],
        help="reward mode per dataset, in DATASETS order; appended to --label",
    )
    ap.add_argument("--head", default="ppo_eval", choices=["ppo_eval", "ppo_explore"])
    ap.add_argument("--num-seeds", type=int, default=3)
    args = ap.parse_args()

    # One shared grid so the three policies are queried at identical observations;
    # span the union of the datasets' log-area ranges.
    ranges = {d: area_range(d) for d in DATASETS}
    lo = max(r[0] for r in ranges.values())
    hi = min(r[1] for r in ranges.values())
    areas = np.linspace(lo, hi, NUM_AREA, dtype=np.float32)
    grid = np.asarray([[a, float(d)] for d in DAYS for a in areas], dtype=np.float32)
    print("log-area ranges per dataset:")
    for d, r in ranges.items():
        print(f"  {d:24s} [{r[0]:+.4f}, {r[1]:+.4f}]")
    print(f"shared grid: {NUM_AREA} areas in [{lo:+.4f}, {hi:+.4f}] x days {DAYS}")

    acts: dict[str, np.ndarray] = {}
    for d in DATASETS:
        per_seed = []
        for s in range(args.num_seeds):
            label = f"{args.label}_{args.mode[DATASETS.index(d)]}"
            ckpt = os.path.join(
                args.run_root, f"plant_plant-data_{d}", label, f"seed_{s}",
                "checkpoint",
            )
            if not os.path.isdir(ckpt):
                print(f"  MISSING {ckpt}")
                continue
            per_seed.append(mean_actions(os.path.abspath(ckpt), args.head, grid))
        acts[d] = np.stack(per_seed)  # (S, G)
        a = acts[d]
        print(
            f"\n{d}: {a.shape[0]} seeds, raw action mean {a.mean():.4f} "
            f"range [{a.min():.4f}, {a.max():.4f}]; "
            f"outside [{ACTION_MIN}, {ACTION_MAX}]: {100 * ((a < ACTION_MIN) | (a > ACTION_MAX)).mean():.1f}% of grid"
        )
        for i, day in enumerate(DAYS):
            sl = acts[d][:, i * NUM_AREA : (i + 1) * NUM_AREA]
            print(
                f"   day {day:2d}: per-seed means "
                + " ".join(f"{v:.4f}" for v in sl.mean(axis=1))
            )

    # Per-day seed statistics. The seed mean's standard error shrinks as 1/sqrt(S),
    # so this is the comparison that actually sharpens with more seeds — unlike a
    # pairwise |seed_i - seed_j| floor, which does not.
    def day_means(a: np.ndarray) -> np.ndarray:
        """(S, G) grid actions → (S, len(DAYS)) per-seed mean action per day."""
        return np.stack(
            [a[:, i * NUM_AREA : (i + 1) * NUM_AREA].mean(axis=1)
             for i in range(len(DAYS))],
            axis=1,
        )

    # Compare the EFFECTIVE action. ClipAction bounds what the env executes and
    # the deployed agent clips the same way, while pi.mean() is an unbounded
    # Gaussian mean — so raw pi.mean() overstates differences in a region no
    # action can actually reach. (Training clips to the data range
    # [0.381, 1.294]; the deployment bounds below are the ops convention.)
    eff = {d: np.clip(a, ACTION_MIN, ACTION_MAX) for d, a in acts.items()}
    dm = {d: day_means(a) for d, a in eff.items()}

    print("\neffective (clipped) mean action by day (seed mean ± SEM):")
    header = "  " + "dataset".ljust(24) + "".join(f"  day {d:<12d}" for d in DAYS)
    print(header)
    for d in DATASETS:
        m, sem = dm[d].mean(0), dm[d].std(0, ddof=1) / np.sqrt(dm[d].shape[0])
        print(
            "  " + d.ljust(24)
            + "".join(f"  {m[i]:.3f}±{sem[i]:.3f}    " for i in range(len(DAYS)))
        )

    # Day-0 → day-14 slope: the absolute reward pays for raw Δlog A, which is
    # largest early, so it should buy light early and coast late. The differential
    # reward subtracts the control's own day-varying growth, which removes that
    # day trend from the reward — so a flatter profile is the signature to look for.
    print("\nday-0 minus day-14 action (how steeply the policy ramps down):")
    for d in DATASETS:
        slope = dm[d][:, 0] - dm[d][:, -1]
        print(
            f"  {d:24s} {slope.mean():+.3f} ± "
            f"{slope.std(ddof=1) / np.sqrt(len(slope)):.3f}"
        )

    def welch(x: np.ndarray, y: np.ndarray) -> float:
        vx = x.var(ddof=1) / len(x)
        vy = y.var(ddof=1) / len(y)
        return float((x.mean() - y.mean()) / np.sqrt(vx + vy))

    print("\nper-day Welch t on the seed means (|t| > ~2.1 ⇒ separated at 10 seeds):")
    for i, a in enumerate(DATASETS):
        for b in DATASETS[i + 1 :]:
            ts = [welch(dm[a][:, k], dm[b][:, k]) for k in range(len(DAYS))]
            print(
                f"  {a:24s} vs {b:24s} "
                + " ".join(f"d{DAYS[k]}:{ts[k]:+.2f}" for k in range(len(DAYS)))
            )
            sa = dm[a][:, 0] - dm[a][:, -1]
            sb = dm[b][:, 0] - dm[b][:, -1]
            print(f"  {'':24s}    ramp-down t = {welch(sa, sb):+.2f}")


if __name__ == "__main__":
    main()
