#!/usr/bin/env python3
"""Two thesis figures for the MountainCar-v0 discussion in Chapter 6 (exploration
with model uncertainty): real-environment state-space dynamics of each
ablation's own data-collecting policy over training, and the information-gain
exploration policy's own in-model PPO return decaying as epistemic
uncertainty is resolved.

Data source: runs/classic_plan_every_cheap_model_long_variants/{task_0001,
task_0011} (MountainCar-v0, "exploit_mean_eig" and "explore_sample_eig"
labels in config.json), 30 seeds each. All real logged tensorboard scalars,
no synthetic data, no fabricated error bars (shading is min/max over the 30
genuine seed replicates).

Key scalars used (see logger.py / main.py):
  - dataset/obs_0: position (state dim 0) of every real-environment transition
    collected by the data-collecting ("behavior"/rollout) policy, one value
    per outer training iteration (steps_per_rollout=1, num_rollouts=1000 for
    this experiment, so index == outer iteration).
  - ppo/explore_return: the in-model PPO training return of the behavior
    policy against its own training objective (task reward for
    exploit_mean_eig's alpha=1/beta=0 config; the information-gain intrinsic
    reward for explore_sample_eig's alpha=0/beta=1 config), logged as ~10
    points per outer iteration.
  - eval/mean_return: task-reward return of the separately-trained evaluation
    policy (always alpha=1/beta=0), used only in the summary printout for
    context in the caption text.

A run that hit its time limit before finishing was resumed as a second
tfevents file in the same seed directory; the later file (by the timestamp
in its filename) is used to override any duplicate iteration indices from an
earlier, truncated attempt.

Usage:
    uv run python3 scripts/plot_mountaincar_eig_dynamics.py \
        --root runs/classic_plan_every_cheap_model_long_variants \
        --exploit-task task_0001 --explore-task task_0011 \
        --out-dynamics /tmp/mountaincar_eig_vs_exploit_dynamics.pdf \
        --out-decay /tmp/mountaincar_eig_explore_return_decay.pdf
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_file_loader


def _extract_value(v):
    if v.HasField("simple_value"):
        return v.simple_value
    if v.HasField("tensor"):
        return float(v.tensor.float_val[0])
    raise ValueError(f"unrecognized value proto for tag {v.tag!r}")


def load_tag(seed_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one scalar tag's (step, value) series for one seed directory.

    Multiple tfevents files (a resumed SLURM job) are merged in filename
    timestamp order, with later files overriding earlier ones at the same
    step -- the earlier file is from a run that timed out before finishing.
    """
    files = sorted(
        glob.glob(os.path.join(seed_dir, "events.out.tfevents.*")),
        key=lambda p: float(os.path.basename(p).split(".")[3]),
    )
    step_to_val: dict[int, float] = {}
    for f in files:
        loader = event_file_loader.EventFileLoader(f)
        for e in loader.Load():
            for v in e.summary.value:
                if v.tag == tag:
                    step_to_val[e.step] = _extract_value(v)
    steps = np.array(sorted(step_to_val.keys()))
    vals = np.array([step_to_val[s] for s in steps])
    return steps, vals


def load_variant(task_dir: str) -> dict[str, np.ndarray]:
    seed_dirs = sorted(glob.glob(os.path.join(task_dir, "seed_*")))
    pos, ret, explore_ret = [], [], []
    for sd in seed_dirs:
        _, p = load_tag(sd, "dataset/obs_0")
        _, r = load_tag(sd, "eval/mean_return")
        _, er = load_tag(sd, "ppo/explore_return")
        pos.append(p)
        ret.append(r)
        explore_ret.append(er)
    lens = {len(p) for p in pos}
    if len(lens) != 1:
        raise RuntimeError(f"ragged dataset/obs_0 lengths across seeds in {task_dir}: {lens}")
    return {
        "pos": np.stack(pos),
        "eval_return": np.stack(ret),
        "explore_return": np.stack(explore_ret),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--exploit-task", default="task_0001")
    ap.add_argument("--explore-task", default="task_0011")
    ap.add_argument("--out-dynamics", required=True)
    ap.add_argument("--out-decay", required=True)
    ap.add_argument("--n-bins", type=int, default=25)
    ap.add_argument("--near-goal", type=float, default=0.45)
    ap.add_argument("--font-size", type=int, default=11)
    ap.add_argument("--serif", action="store_true", default=True)
    args = ap.parse_args()

    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": args.font_size}
    if args.serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
    plt.rcParams.update(rc)

    exploit = load_variant(os.path.join(args.root, args.exploit_task))
    explore = load_variant(os.path.join(args.root, args.explore_task))

    exploit_pos, explore_pos = exploit["pos"], explore["pos"]
    n_iters = exploit_pos.shape[1]
    assert explore_pos.shape[1] == n_iters

    # ---------------- Figure 1: goal-discovery dynamics ----------------
    n_bins = args.n_bins
    bin_size = n_iters // n_bins
    bin_centers = (np.arange(n_bins) + 0.5) * bin_size

    def binned_max(pos):
        return np.stack(
            [pos[:, i * bin_size : (i + 1) * bin_size].max(axis=1) for i in range(n_bins)],
            axis=1,
        )

    exploit_binned = binned_max(exploit_pos)
    explore_binned = binned_max(explore_pos)
    exploit_frac = (exploit_binned >= args.near_goal).mean(axis=0)
    explore_frac = (explore_binned >= args.near_goal).mean(axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(
        bin_centers, explore_frac, label="Information-gain exploration policy",
        color="#0072B2", linewidth=2, marker="o", markersize=3,
    )
    ax.plot(
        bin_centers, exploit_frac, label="Task-reward-only policy (model-based PPO)",
        color="#D55E00", linewidth=2, marker="o", markersize=3,
    )
    ax.set_xlabel("Training iteration (real environment steps)")
    ax.set_ylabel(
        f"Fraction of {exploit_pos.shape[0]} seeds reaching\n"
        f"position $\\geq${args.near_goal} ({bin_size}-iteration windows)"
    )
    ax.set_xlim(0, n_iters)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(args.out_dynamics)
    plt.close(fig)

    # ---------------- Figure 2: explore-return decay ----------------
    explore_ret = explore["explore_return"]
    n_pts_per_iter = explore_ret.shape[1] // n_iters
    per_iter = explore_ret.reshape(explore_ret.shape[0], n_iters, n_pts_per_iter)[:, :, -1]
    per_bin = np.stack(
        [per_iter[:, i * bin_size : (i + 1) * bin_size].mean(axis=1) for i in range(n_bins)],
        axis=1,
    )
    mean2, lo2, hi2 = per_bin.mean(axis=0), per_bin.min(axis=0), per_bin.max(axis=0)

    fig2, ax2 = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax2.plot(bin_centers, mean2, color="#0072B2", linewidth=2)
    ax2.fill_between(bin_centers, lo2, hi2, color="#0072B2", alpha=0.2, linewidth=0)
    ax2.set_xlabel("Training iteration (real environment steps)")
    ax2.set_ylabel("In-model PPO return\n(information-gain bonus reward)")
    ax2.set_xlim(0, n_iters)
    ax2.set_yscale("log")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    fig2.savefig(args.out_decay)
    plt.close(fig2)

    # ---------------- context numbers for the caption/prose ----------------
    print(f"exploit eval/mean_return, last 100 iters mean: {exploit['eval_return'][:, -100:].mean():.1f}")
    print(f"explore eval/mean_return, last 100 iters mean: {explore['eval_return'][:, -100:].mean():.1f}")
    print(f"explore_return first bin mean: {mean2[0]:.2f}, last bin mean: {mean2[-1]:.2f}")
    print(f"seeds with declining explore_return (first bin > last bin): "
          f"{(per_bin[:, 0] > per_bin[:, -1]).sum()}/{per_bin.shape[0]}")


if __name__ == "__main__":
    main()
