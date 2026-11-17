#!/usr/bin/env python3
"""Compare final return across seeds: model-free PPO vs M-PPO, on Pendulum-v1.

PPO (``main_model_free.py``, ``--ppo.num-envs 1``) trains directly on the real
env; M-PPO (``main.py``) plans against a learned world model and only takes a
handful of real env steps. Both log ``eval/mean_return`` (held-out policy
eval) keyed by cumulative real env steps in TensorBoard, at
``<root>/seed_<seed>/events.out.tfevents*`` — this takes each run's last
logged point, plots the two seed-distributions side by side, and reports the
real-env-step budget each spent to get there.
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


def _final_scalar(run_dir: Path, tag: str) -> tuple[int, float] | None:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    items = ea.Scalars(tag)
    return items[-1].step, items[-1].value


def _load(root: Path, label: str) -> pd.DataFrame:
    rows = []
    for seed_dir in _seed_dirs(root):
        seed = int(seed_dir.name.removeprefix("seed_"))
        result = _final_scalar(seed_dir, TAG)
        if result is None:
            continue
        step, value = result
        rows.append({"method": label, "seed": seed, "real_steps": step, "return": value})
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
    parser.add_argument("--mppo-label", default="M-PPO")
    parser.add_argument("--ppo-label", default="PPO")
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_vs_mppo_pendulum.png"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--font-size", type=int, default=None)
    args = parser.parse_args()

    ppo_df = _load(args.ppo_root, args.ppo_label)
    mppo_df = _load(args.mppo_root, args.mppo_label)
    df = pd.concat([ppo_df, mppo_df], ignore_index=True)

    step_budget = df.groupby("method")["real_steps"].max()
    order = [args.ppo_label, args.mppo_label]
    tick_labels = [f"{m}\n({step_budget[m]:,} real steps)" for m in order]

    print(
        df.groupby("method")["return"]
        .agg(n="count", mean="mean", std="std")
        .reindex(order)
        .to_string()
    )

    rc: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if args.font_size is not None:
        rc["font.size"] = args.font_size
    sns.set_theme(style="white", rc=rc)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
    palette = {args.ppo_label: "#4C72B0", args.mppo_label: "#DD8452"}
    sns.boxplot(
        data=df,
        x="method",
        y="return",
        order=order,
        palette=palette,
        width=0.5,
        fliersize=0,
        boxprops={"alpha": 0.4},
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="method",
        y="return",
        order=order,
        palette=palette,
        size=5,
        alpha=0.8,
        jitter=0.15,
        linewidth=0.5,
        edgecolor="white",
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    ax.set_ylabel("held-out eval return")
    ax.set_title(f"Pendulum-v1 — {len(ppo_df)} vs {len(mppo_df)} seeds")
    sns.despine(ax=ax)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
