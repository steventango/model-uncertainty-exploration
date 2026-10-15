"""Compute a masked-reward Zone-11 baseline and its surplus range, pinned to one
experiment, from a plant-data parquet plus its minari training transitions.

The masked modes gate the resulting quantity against ``_MASK_F`` times a per-day
Zone-11 baseline. Which experiment that zone belongs to matters a lot: E21 Z11's
cohort outgrew E18 Z11's by roughly 56% by day 14, so a 90% gate against E21 is a
much harder target than the same gate against E18.

Mirrors plant-data ``join_zones._zone11_baseline``: per-day mean over the
baseline zone's valid rows (``valid_area = clean_area > 0``), linearly
interpolated across days with no measurement, with day 0 filled from day 1 so a
day-0 lookup is defined.

Note the baseline is NOT invariant to which experiments are in the join:
``transform_outlier_detection`` runs dataset-wide, so adding an experiment shifts
the surviving rows and hence the per-day means. Always regenerate from the same
parquet the policy will be trained on rather than reusing constants from another
version — the E18 constants in plant_env.py came from the v28
mixed-e18-e19-e20 parquet and differ from v29's by up to 0.048 (~1.6%).

Usage:
    uv run python scripts/compute_masked_baseline.py \
        --parquet /data/plant-rl/offline/v29/e18e21-daily-v29.parquet \
        --experiment 21 --zone 11 \
        --dataset plant-data/e18e21-daily-v29
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offline_data import load_offline_transitions  # noqa: E402
from plant_env import _MASK_F  # noqa: E402

NUM_DAYS = 15


def _dedupe(parquet: str) -> pl.DataFrame:
    """Read the parquet, keeping the first of any duplicated column name."""
    names = pl.scan_parquet(parquet).collect_schema().names()
    seen: set[str] = set()
    keep: list[int] = []
    for i, c in enumerate(names):
        if c not in seen:
            seen.add(c)
            keep.append(i)
    return pl.read_parquet(parquet)[:, keep]


def baseline(df: pl.DataFrame, experiment: int, zone: int, col: str) -> np.ndarray:
    rows = df.filter(
        (pl.col("experiment") == experiment)
        & (pl.col("zone") == zone)
        & pl.col("clean_area").is_not_null()
        & (pl.col("clean_area") > 0)
        & pl.col(col).is_not_null()
    )
    if rows.is_empty():
        raise SystemExit(f"no valid rows for E{experiment} Z{zone} / {col}")
    days = pl.DataFrame({"day": list(range(NUM_DAYS))}).with_columns(
        pl.col("day").cast(df["day"].dtype)
    )
    agg = rows.group_by("day").agg(pl.col(col).mean().alias("v"))
    s = (
        days.join(agg, on="day", how="left")
        .sort("day")
        .with_columns(pl.col("v").interpolate())
    )
    v = np.asarray(s["v"].to_list(), dtype=float)
    if not np.isfinite(v[0]) and np.isfinite(v[1]):
        v[0] = v[1]
    return v


def _fmt(name: str, v: np.ndarray) -> str:
    body = "\n".join(f"        {float(x)!r}," for x in v)
    return f"{name} = jnp.asarray(\n    [\n{body}\n    ],\n    dtype=jnp.float32,\n)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--experiment", type=int, required=True)
    ap.add_argument("--zone", type=int, default=11)
    ap.add_argument("--dataset", required=True, help="minari id for the surplus range")
    ap.add_argument("--suffix", default=None, help="constant name suffix, e.g. E21")
    args = ap.parse_args()
    suffix = args.suffix or f"E{args.experiment}"

    df = _dedupe(args.parquet)
    area = baseline(df, args.experiment, args.zone, "clean_area")
    growth = baseline(df, args.experiment, args.zone, "area_reward")

    print(f"# Pinned to E{args.experiment} Z{args.zone}, {Path(args.parquet).name}, "
          f"f={_MASK_F}")
    print(_fmt(f"_BASELINE_AREA_{suffix}", area))
    print(_fmt(f"_BASELINE_GROWTH_{suffix}", growth))

    # Surplus range over the training transitions, under THIS baseline. Both
    # modes gate the resulting quantity against the next day's baseline, so the
    # bounds are measured over next-obs off-target rows (day-0 starts excluded,
    # since a start is never a resulting area).
    tr = load_offline_transitions(args.dataset, include_time=True)[0]
    obs = np.asarray(tr.obs)
    next_obs = np.asarray(tr.info["next_obs"])
    next_day = np.clip(np.rint(next_obs[:, 1]).astype(int), 0, NUM_DAYS - 1)

    thr = _MASK_F * area[next_day]
    off = np.exp(next_obs[:, 0]) < thr
    s_log = (next_obs[:, 0] - np.log(thr))[off]
    print(f"\n# off-target (log) {int(off.sum())}/{off.size} "
          f"= {100 * off.mean():.1f}% of transitions")
    print(
        f"_SURPLUS_LOG_MIN_{suffix}, _SURPLUS_LOG_MAX_{suffix} = "
        f"{float(s_log.min())!r}, {float(s_log.max())!r}"
    )

    g = next_obs[:, 0] - obs[:, 0]
    thr_g = _MASK_F * growth[next_day]
    off_g = g < thr_g
    s_g = (g - thr_g)[off_g]
    print(f"# off-target (growth) {int(off_g.sum())}/{off_g.size} "
          f"= {100 * off_g.mean():.1f}%")
    print(
        f"_SURPLUS_GROWTH_MIN_{suffix}, _SURPLUS_GROWTH_MAX_{suffix} = "
        f"{float(s_g.min())!r}, {float(s_g.max())!r}"
    )


if __name__ == "__main__":
    main()
