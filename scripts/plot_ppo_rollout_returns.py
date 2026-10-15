#!/usr/bin/env python3
"""Plot in-model PPO training returns for a single variant's two co-trained
configs: the "explore" config (the variant's own alpha/beta/bonus, drives real
data collection) vs the "eval" config (always alpha=1, beta=0 — oracle reward,
no intrinsic bonus; trained on the same model, deployed to the real env for
`eval/mean_return`).

Both are logged per PPO update within an outer (real-step) iteration, at step
= outer * ppo_timesteps + update_idx * env_steps_per_update (main.py's
log_ppo_returns). Reduced here to one point per outer iteration — its last
(most-converged) logged update — so it's comparable to the other per-real-step
learning curves.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_learning_curves import (  # noqa: E402
    ENVS,
    _latest_events_file,
    _style_axis,
    discover_run_entries,
)

TAGS = ("ppo/explore_return", "ppo/eval_return")
TAG_LABEL = {
    "ppo/explore_return": "explore (bonus policy, in-model)",
    "ppo/eval_return": "eval (oracle policy, in-model)",
}


def _load_run(run_dir: Path, meta: dict, ppo_timesteps: float) -> list[dict]:
    events_file = _latest_events_file(run_dir)
    if events_file is None:
        return []
    ea = EventAccumulator(str(events_file))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    rows: list[dict] = []
    for tag in TAGS:
        if tag not in available:
            continue
        # Keep only the last (most-converged) logged point per outer
        # iteration — each iteration retrains PPO from scratch over
        # ppo_timesteps env-steps, so earlier points are mid-training, not
        # comparable across iterations.
        last_by_iter: dict[int, float] = {}
        last_step_by_iter: dict[int, int] = {}
        for scalar in ea.Scalars(tag):
            outer = int(scalar.step // ppo_timesteps)
            if outer not in last_step_by_iter or scalar.step > last_step_by_iter[outer]:
                last_step_by_iter[outer] = scalar.step
                last_by_iter[outer] = scalar.value
        rows.extend(
            {
                **meta,
                "config": TAG_LABEL[tag],
                "dataset_size": outer + 1,
                "value": value,
            }
            for outer, value in last_by_iter.items()
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot in-model PPO training returns (explore vs eval config) for "
            "one variant's task_*/config.json runs."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--variant", required=True, help="config.json label to select, e.g. explore_sample_eig"
    )
    parser.add_argument(
        "--ppo-timesteps",
        type=float,
        default=1e7,
        help="PPOConfig.total_timesteps used for the run (default: 1e7, this repo's default)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--font-size", type=int, default=None)
    parser.add_argument("--serif", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    entries = discover_run_entries(args.root.resolve())
    entries = [(d, m) for d, m in entries if m["variant"] == args.variant]
    if not entries:
        raise RuntimeError(f"no runs found for variant {args.variant!r} under {args.root}")

    rows: list[dict] = []
    for run_dir, meta in entries:
        rows.extend(_load_run(run_dir, meta, args.ppo_timesteps))
    if not rows:
        raise RuntimeError(
            f"no {TAGS} data found for variant {args.variant!r} under {args.root}"
        )
    df = pd.DataFrame(rows)

    if args.smooth_window != 0:
        df = df.sort_values(["env", "config", "seed", "dataset_size"])
        window = args.smooth_window
        df["value"] = df.groupby(["env", "config", "seed"], sort=False)["value"].transform(
            lambda s: s.ewm(
                alpha=2.0 / ((window or max(3, int(len(s) * 0.1))) + 1), adjust=False
            ).mean()
        )

    envs_present = [env for env in ENVS if env in df["env"].unique()]
    envs_present.extend(sorted(set(df["env"].unique()) - set(ENVS)))
    df["env"] = pd.Categorical(df["env"], categories=envs_present, ordered=True)
    hue_order = [TAG_LABEL[t] for t in TAGS if TAG_LABEL[t] in df["config"].unique()]
    df["config"] = pd.Categorical(df["config"], categories=hue_order, ordered=True)

    rc: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if args.font_size is not None:
        rc["font.size"] = args.font_size
    if args.serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
    sns.set_theme(style="white", rc=rc)

    # Column width follows --scale to match the multi-row plots' per-env
    # width, but row height does NOT: this is a single row, so unlike the
    # 3-row grid it has no vertical budget pressure, and needs enough
    # absolute height for a long rotated y-axis label to fit at --font-size
    # without clipping (a rotated label's required space scales with string
    # length, not just line height).
    panel_width = (2.8 * args.scale) * 1.15
    panel_height = 3.2
    legend_row_height = 0.5
    n_envs = len(envs_present)
    fig = plt.figure(
        figsize=(n_envs * panel_width, panel_height + legend_row_height),
        constrained_layout=True,
    )
    # Legend gets its own gridspec row (not a floating fig.legend) so
    # constrained_layout reserves space for it instead of overlapping the
    # bottom row's x-axis labels.
    gs = fig.add_gridspec(2, n_envs, height_ratios=[1, legend_row_height])

    legend_handles = legend_labels = None
    for idx, env in enumerate(envs_present):
        ax = fig.add_subplot(gs[0, idx])
        env_df = df[df["env"] == env]
        sns.lineplot(
            data=env_df,
            x="dataset_size",
            y="value",
            hue="config",
            hue_order=hue_order,
            errorbar=("ci", 95),
            n_boot=args.n_boot,
            ax=ax,
            legend=idx == 0,
        )
        for line in ax.lines:
            line.set_alpha(0.85)
        _style_axis(ax)
        ax.set_title(env, pad=12)
        ax.set_xlabel("Time step")
        ax.set_ylabel("In-model PPO return" if idx == 0 else "")
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    if legend_handles is not None:
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=len(hue_order),
            frameon=False,
            columnspacing=1.2,
            handlelength=1.8,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" recomputes an even-tighter bbox on top of
    # constrained_layout's own solve, which can clip the rotated y-axis
    # label; constrained_layout already guarantees everything fits inside
    # figsize, so save as-is.
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
