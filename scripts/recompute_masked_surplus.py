"""Recompute the off-target surplus normalization constants for the masked
reward modes under next-area gating.

``masked_log`` evaluates the resulting area (next_obs) against the next day's
Zone-11 baseline, in log-area space. The off-target penalty min-max-normalizes
the surplus (qty - f*baseline) over off-target rows to [-1, 0]. Those min/max
bounds must therefore be measured over the *next-obs* off-target transitions
(which exclude each trajectory's day-0 start, since day 0 is never a resulting
area).

``masked_growth`` instead gates each step's *growth* (next_log_area - log_area)
against the next day's growth baseline, so its bounds are measured over the
same next-obs off-target transitions as ``masked_log``.

Prints the values to paste into plant_env.py's _SURPLUS_LOG_* / _SURPLUS_GROWTH_*.

Usage:
    uv run python scripts/recompute_masked_surplus.py \
        --npz /path/to/masked/offline_transitions.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plant_env import _BASELINE_AREA, _BASELINE_GROWTH, _MASK_F  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        required=True,
        help="masked offline_transitions.npz (obs_dim 2 = [log_area, time])",
    )
    args = parser.parse_args()

    base = np.asarray(_BASELINE_AREA)
    base_growth = np.asarray(_BASELINE_GROWTH)
    npz = np.load(args.npz)
    obs = npz["obs"]
    next_obs = npz["next_obs"]
    if next_obs.shape[1] < 2:
        raise SystemExit("npz must be a masked (obs_dim 2) export with a time column")

    next_log_area = next_obs[:, 0]
    next_day = np.clip(np.rint(next_obs[:, 1]).astype(int), 0, len(base) - 1)
    threshold = _MASK_F * base[next_day]
    next_area = np.exp(next_log_area)
    off = next_area < threshold

    surplus_log = (next_log_area - np.log(threshold))[off]
    print(f"off-target next-obs transitions: {int(off.sum())} / {off.size}")
    print(
        "_SURPLUS_LOG_MIN, _SURPLUS_LOG_MAX = "
        f"{float(surplus_log.min())!r}, {float(surplus_log.max())!r}"
    )

    # masked_growth gates growth against the *next* day's growth baseline.
    growth = next_log_area - obs[:, 0]
    threshold_growth = _MASK_F * base_growth[next_day]
    off_growth = growth < threshold_growth
    surplus_growth = (growth - threshold_growth)[off_growth]
    print(
        f"off-target growth transitions: {int(off_growth.sum())} / {off_growth.size}"
    )
    print(
        "_SURPLUS_GROWTH_MIN, _SURPLUS_GROWTH_MAX = "
        f"{float(surplus_growth.min())!r}, {float(surplus_growth.max())!r}"
    )


if __name__ == "__main__":
    main()
