#!/usr/bin/env python3
"""Learning curve: model-free PPO vs M-PPO, on Pendulum-v1.

PPO (``main_model_free.py``, ``--ppo.num-envs 1``) trains directly on the real
env; M-PPO (``main.py``) plans against a learned world model and only takes a
handful of real env steps. Both log ``eval/mean_return`` (held-out policy
eval) keyed by cumulative real env steps in TensorBoard, at
``<root>/seed_<seed>/events.out.tfevents*`` — this loads the full per-seed
time series (not just the final point) for both and plots mean +/- 95% CI
across seeds against real env steps, on a log x-axis so PPO's 100k-step curve
and M-PPO's 1k-step curve are both readable on one axis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "eval/mean_return"


def _seed_dirs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )


def _curve(run_dir: Path, tag: str) -> list[tuple[int, float]]:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def _load(root: Path, label: str) -> pd.DataFrame:
    rows = []
    for seed_dir in _seed_dirs(root):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for step, value in _curve(seed_dir, TAG):
            rows.append(
                {"method": label, "seed": seed, "real_steps": step, "return": value}
            )
    if not rows:
        raise RuntimeError(f"no {TAG!r} scalars found under {root}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-root",
        type=Path,
        default=Path("runs/model_free_ppo_pendulum_30seed"),
        help="main_model_free.py output dir (seed_* subdirs)",
    )
    parser.add_argument(
        "--mppo-root",
        type=Path,
        default=Path("runs/classic_plan_every_cheap_model_long_variants/task_0000"),
        help="main.py output dir (seed_* subdirs), e.g. the exploit_mean_eig/Pendulum-v1 task",
    )
    parser.add_argument("--ppo-label", default="PPO")
    parser.add_argument("--mppo-label", default="M-PPO")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/ppo_vs_mppo_learning_curve.png")
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--font-size", type=int, default=None)
    args = parser.parse_args()

    ppo_df = _load(args.ppo_root, args.ppo_label)
    mppo_df = _load(args.mppo_root, args.mppo_label)
    df = pd.concat([ppo_df, mppo_df], ignore_index=True)
    # log x-axis can't render real_steps=0 (the pre-training eval point);
    # nudge it to 1 so the curves still start from the untrained baseline.
    df["real_steps"] = df["real_steps"].clip(lower=1)

    hue_order = [args.ppo_label, args.mppo_label]
    n_seeds = {m: df.loc[df["method"] == m, "seed"].nunique() for m in hue_order}

    rc: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if args.font_size is not None:
        rc["font.size"] = args.font_size
    sns.set_theme(style="white", rc=rc)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    sns.lineplot(
        data=df,
        x="real_steps",
        y="return",
        hue="method",
        hue_order=hue_order,
        palette={args.ppo_label: "#4C72B0", args.mppo_label: "#DD8452"},
        errorbar=("ci", 95),
        n_boot=args.n_boot,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("real env steps")
    ax.set_ylabel("held-out eval return")
    ax.set_title(
        f"Pendulum-v1 — {args.ppo_label} ({n_seeds[args.ppo_label]} seeds) vs "
        f"{args.mppo_label} ({n_seeds[args.mppo_label]} seeds)"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=None, frameon=False)
    sns.despine(ax=ax)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
