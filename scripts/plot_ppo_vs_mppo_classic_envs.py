#!/usr/bin/env python3
"""PPO vs M-PPO return, one column per classic-control env.

Reproduces the layout of the top ("Return") row of
``runs/compare_plan_every/G_cheap_model_long_variants.pdf``
(``scripts/plot_learning_curves.py``'s style) but with two curves
(model-free PPO vs M-PPO) instead of four ablation variants, and a linear
(not log) x-axis.

PPO (``main_model_free.py``, ``--ppo.num-envs 1``) trains directly on the
real env; M-PPO (``main.py``) plans against a learned world model and only
takes a handful of real env steps. Both log ``eval/mean_return`` (held-out
policy eval) keyed by cumulative real env steps in TensorBoard, at
``<root>/seed_<seed>/events.out.tfevents*``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "eval/mean_return"

CLASSIC_ENVS = (
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
)


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


def _load(root: Path, env: str, method: str) -> pd.DataFrame:
    rows = []
    for seed_dir in _seed_dirs(root):
        seed = int(seed_dir.name.removeprefix("seed_"))
        for step, value in _curve(seed_dir, TAG):
            rows.append(
                {
                    "env": env,
                    "method": method,
                    "seed": seed,
                    "real_steps": step,
                    "return": value,
                }
            )
    if not rows:
        raise RuntimeError(f"no {TAG!r} scalars found under {root}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-root-template",
        default="runs/model_free_ppo_{env}_30seed",
        help="main_model_free.py output dir per env, {env} is substituted",
    )
    parser.add_argument(
        "--mppo-root",
        type=Path,
        default=Path("runs/classic_plan_every_cheap_model_long_variants"),
        help="main.py sweep root (task_*/config.json layout)",
    )
    parser.add_argument("--mppo-label", default="exploit_mean_eig")
    parser.add_argument("--ppo-label", default="PPO")
    parser.add_argument("--mppo-display-label", default="M-PPO")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/ppo_vs_mppo_classic_envs.pdf")
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--font-size", type=int, default=None)
    parser.add_argument("--serif", action="store_true")
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="log-scale the x-axis (real env steps span 1000 for M-PPO vs "
        "100000 for PPO, so linear compresses M-PPO's whole curve into a sliver)",
    )
    args = parser.parse_args()

    ppo_root = Path(str(args.ppo_root_template).format(env="Pendulum-v1"))
    if not ppo_root.parent.exists():
        raise RuntimeError(f"template resolves oddly, check --ppo-root-template")

    # Map task_NNNN dirs -> env via config.json, restricted to --mppo-label.
    mppo_dirs: dict[str, Path] = {}
    for task_dir in sorted(args.mppo_root.iterdir()):
        cfg_path = task_dir / "config.json"
        if not cfg_path.is_file():
            continue
        cfg = __import__("json").loads(cfg_path.read_text())
        if cfg.get("label") == args.mppo_label:
            mppo_dirs[cfg["env"]] = task_dir

    frames = []
    for env in CLASSIC_ENVS:
        ppo_dir = Path(str(args.ppo_root_template).format(env=env))
        frames.append(_load(ppo_dir, env, args.ppo_label))
        mppo_dir = mppo_dirs.get(env)
        if mppo_dir is None:
            raise RuntimeError(
                f"no M-PPO task with label={args.mppo_label!r} for env={env!r} "
                f"under {args.mppo_root}"
            )
        frames.append(_load(mppo_dir, env, args.mppo_display_label))
    df = pd.concat(frames, ignore_index=True)
    if args.log_x:
        # log x-axis can't render real_steps=0 (the pre-training eval point).
        df["real_steps"] = df["real_steps"].clip(lower=1)

    hue_order = [args.ppo_label, args.mppo_display_label]
    palette = {args.ppo_label: "#4C72B0", args.mppo_display_label: "#DD8452"}

    rc: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if args.font_size is not None:
        rc["font.size"] = args.font_size
    if args.serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
    sns.set_theme(style="white", rc=rc)

    n_envs = len(CLASSIC_ENVS)
    panel_width, panel_height = 2.8, 2.6
    legend_row_height = 0.5
    fig = plt.figure(
        figsize=(n_envs * panel_width, panel_height + legend_row_height),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(2, n_envs, height_ratios=[1, legend_row_height])

    legend_handles = legend_labels = None
    for col, env in enumerate(CLASSIC_ENVS):
        ax = fig.add_subplot(gs[0, col])
        env_df = df[df["env"] == env]
        sns.lineplot(
            data=env_df,
            x="real_steps",
            y="return",
            hue="method",
            hue_order=hue_order,
            palette=palette,
            errorbar=("ci", 95),
            n_boot=args.n_boot,
            ax=ax,
            legend=col == 0,
        )
        if args.log_x:
            ax.set_xscale("log")
        ax.set_title(env)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Return" if col == 0 else "")
        sns.despine(ax=ax)
        if col == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=len(hue_order),
        frameon=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
