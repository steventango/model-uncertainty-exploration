"""Export offline plant transitions to a self-contained .npz for deployment.

The plant-rl deployment host has neither minari nor the dataset registry, so
the nightly-retraining agent (AdaptivePPOPolicy) loads its initial replay
buffer from this file. Metadata mirrors what main.py derives from the dataset
(PlantEnvParams bounds and episode length).

Usage (from the repo root):
    uv run python scripts/export_offline_transitions.py \
        --reward-mode masked_log --out offline_transitions.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offline_data import load_offline_transitions  # noqa: E402
from plant_env import PlantEnv, reward_mode_requires_time  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="plant-data/visu-v2-v27")  # noqa: E501
    parser.add_argument(
        "--reward-mode",
        default="analytic",
        choices=["area", "analytic", "analytic_diff", "masked_log", "masked_growth"],
        help="Determines include_time (masked/diff modes export [log_area, day] "
        "obs) and the dynamics target (analytic_diff exports the de-confounded "
        "growth advantage instead of the observation delta).",
    )
    parser.add_argument(
        "--include-time",
        action="store_true",
        help="Force the day into the obs ([log_area, day]). Masked and "
        "differential modes force it on regardless; pass it for other modes when "
        "the checkpoint being shipped was trained with --include-time, or the "
        "buffer's obs_dim will not match the policy's.",
    )
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    include_time = args.include_time or reward_mode_requires_time(args.reward_mode)
    # A throwaway env just to ask the reward mode what the dynamics target is.
    probe = PlantEnv(
        1,
        reward_mode=args.reward_mode,
        include_time=include_time,
        ctl_growth=np.zeros(1),
        ctl_energy=np.ones(1),
    )
    (
        transitions,
        action_space,
        observation_space,
        max_ep_len,
        ctl_growth,
        ctl_energy,
    ) = load_offline_transitions(
        args.dataset, include_time=include_time, dyn_target=probe.dyn_target
    )

    obs_all = np.concatenate(
        [np.asarray(transitions.obs), np.asarray(transitions.info["next_obs"])], axis=0
    )
    area_lo, area_hi = float(obs_all[:, 0].min()), float(obs_all[:, 0].max())
    act_arr = np.asarray(transitions.action)
    act_lo, act_hi = act_arr.min(axis=0), act_arr.max(axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        obs=np.asarray(transitions.obs, dtype=np.float32),
        action=np.asarray(transitions.action, dtype=np.float32),
        reward=np.asarray(transitions.reward, dtype=np.float32),
        next_obs=np.asarray(transitions.info["next_obs"], dtype=np.float32),
        # Dynamics-model regression target. Equals next_obs - obs for
        # dyn_target="delta_obs"; the de-confounded growth advantage otherwise.
        dyn_target=np.asarray(transitions.info["dyn_target"], dtype=np.float32),
        # Per-day control tables. The differential setup needs them to turn a
        # predicted advantage back into an absolute area, and for the energy
        # reference in the reward. Shipped with the buffer so the deployment host
        # needs neither minari nor the parquet.
        ctl_growth=np.asarray(ctl_growth, dtype=np.float32),
        ctl_energy=np.asarray(ctl_energy, dtype=np.float32),
        terminated=np.asarray(transitions.terminated, dtype=bool),
        truncated=np.asarray(transitions.truncated, dtype=bool),
        # Bounds from the loaded transitions, matching main.py. plant-data
        # derives the Minari action space from pre-filter stats, so its declared
        # low is 0.00276 — an off/night row no episode contains. Shipping that
        # would let the nightly retrain's ClipAction explore two orders of
        # magnitude below anything observed.
        area_min=np.float32(area_lo),
        area_max=np.float32(area_hi),
        act_low=np.asarray(act_lo, dtype=np.float32),
        act_high=np.asarray(act_hi, dtype=np.float32),
        max_ep_len=np.int32(max_ep_len),
        include_time=np.bool_(include_time),
        dataset=np.str_(args.dataset),
        reward_mode=np.str_(args.reward_mode),
    )
    n, obs_dim = transitions.obs.shape
    print(
        f"Wrote {out} | N={n} obs_dim={obs_dim} "
        f"area=[{area_lo:.4f}, {area_hi:.4f}] "
        f"act=[{float(act_lo[0]):.4f}, {float(act_hi[0]):.4f}] "
        f"max_ep_len={max_ep_len}"
    )


if __name__ == "__main__":
    main()
