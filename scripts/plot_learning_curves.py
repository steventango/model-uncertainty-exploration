#!/usr/bin/env python3
"""Plot learning curves from experiment TensorBoard logs (task_*/config.json layout)."""

import argparse
import functools
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.algorithms as _sns_algo
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

jax.config.update("jax_platform_name", "cpu")

# ---------------------------------------------------------------------------
# JAX-accelerated bootstrap — monkeypatched into seaborn.algorithms.bootstrap
# ---------------------------------------------------------------------------
_original_bootstrap = _sns_algo.bootstrap
_jax_rng = jax.random.key(42)


@functools.lru_cache(maxsize=16)
def _jax_mean_boot_fn(n_boot: int):
    """Return a JIT-compiled (n_boot, n_seeds) → (n_boot,) mean-bootstrap fn."""

    @jax.jit
    def _fn(data: jax.Array, key: jax.Array) -> jax.Array:
        n = data.shape[0]
        idx = jax.random.randint(key, (n_boot, n), 0, n)
        return data[idx].mean(axis=1)  # (n_boot,)

    return _fn


def _jax_bootstrap(*args, **kwargs):
    """Drop-in replacement for seaborn.algorithms.bootstrap using JAX.

    Vectorises the n_boot resample loop into a single matrix op.
    Falls back to the original for units-based or non-mean cases.
    """
    global _jax_rng
    n_boot = int(kwargs.get("n_boot", 10000))
    func = kwargs.get("func", "mean")
    units = kwargs.get("units", None)

    # Only accelerate the common mean/nanmean case (seaborn lineplot default).
    _is_mean = func in ("mean", "nanmean") or func is np.mean or func is np.nanmean
    if units is not None or not _is_mean:
        return _original_bootstrap(*args, **kwargs)

    data = jnp.asarray(args[0], dtype=jnp.float32)
    _jax_rng, key = jax.random.split(_jax_rng)
    return np.asarray(_jax_mean_boot_fn(n_boot)(data, key))


# Patch both the module attribute and the reference inside seaborn._statistics.
_sns_algo.bootstrap = _jax_bootstrap
try:
    import seaborn._statistics as _sns_stats

    # _statistics imports algorithms as `algo`; patch its reference too.
    if hasattr(_sns_stats, "algo"):
        _sns_stats.algo.bootstrap = _jax_bootstrap
except Exception:
    pass

ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
]

EXPLORE_POLICY = {
    "a1p0_b0p0": "exploit",
    "a0p0_b1p0": "explore",
    "a1p0_b1p0": "both",
}

# std variants first, then eig variants, in the same policy/mode order
VARIANT_ORDER = [
    "exploit/mean/std",
    "exploit/sample/std",
    "explore/mean/std",
    "explore/sample/std",
    "both/mean/std",
    "both/sample/std",
    "exploit/mean/eig",
    "exploit/sample/eig",
    "explore/mean/eig",
    "explore/sample/eig",
    "both/mean/eig",
    "both/sample/eig",
]

RUN_GROUP_COLS = ["env", "variant", "seed"]
DEFAULT_SMOOTH_WINDOW_FRAC = 0.1
MIN_SMOOTH_WINDOW = 3

PlotConfig = dict[str, str | bool | Callable[[pd.Series], pd.Series]]

PLOTS: dict[str, PlotConfig] = {
    "eval/mean_return": {
        "value_col": "mean_return",
        "ylabel": "Return",
    },
    "validation/dynamics_mae": {
        "value_col": "dynamics_mae",
        "ylabel": "Dynamics MAE",
        "yscale": "log",
    },
    "validation/reward_mae": {
        "value_col": "reward_mae",
        "ylabel": "Reward MAE",
        "yscale": "log",
    },
    "validation/termination_f1": {
        "source_col": "termination_f1",
        "value_col": "termination_f1_loss",
        "ylabel": "Termination F1 Loss",
        "transform": lambda series: 1.0 - series,
        "ylim_bottom": 0.0,
    },
    "validation/mean_uncertainty": {
        "value_col": "mean_uncertainty",
        "ylabel": "Uncertainty",
        "yscale": "log",
    },
    "time/model_train": {
        "value_col": "model_train",
        "ylabel": "Model Train Time (s)",
        "yscale": "log",
    },
    "time/ppo_train_s": {
        "value_col": "ppo_train_s",
        "ylabel": "PPO Train Time (s)",
        "yscale": "log",
    },
    "sweep/chosen_length_scale": {
        "value_col": "chosen_length_scale",
        "ylabel": "Length Scale",
    },
}


@lru_cache
def _env_steps_per_iteration(env_name: str) -> int:
    """Env steps collected per outer iteration (matches main.py rollout config)."""
    import gymnax

    _, env_params = gymnax.make(env_name)
    return env_params.max_steps_in_episode // 10


def _rescale_iteration_to_env_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Map validation iteration index to cumulative env steps in the dataset."""
    parts = []
    for env, group in df.groupby("env", sort=False):
        group = group.copy()
        steps = _env_steps_per_iteration(str(env))
        group["dataset_size"] = (group["dataset_size"] + 1) * steps
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _load_col_for_cfg(plot_cfg: PlotConfig) -> str:
    source_col = plot_cfg.get("source_col")
    if isinstance(source_col, str):
        return source_col
    value_col = plot_cfg["value_col"]
    assert isinstance(value_col, str)
    return value_col


def _load_col_for_tag(tag: str) -> str:
    return _load_col_for_cfg(PLOTS[tag])


DEFAULT_COMBINED_OUTPUT = "learning_curves.png"
PANEL_HEIGHT = 2.8
PANEL_ASPECT = 1.15
ROW_PAD = 0.48
COL_PAD = 0.38
LEGEND_ROW_HEIGHT = 0.2
TITLE_PAD = 12
SAVE_PAD_INCHES = 0.2


def _auto_smooth_window(n_points: int, window: int | None) -> int:
    if window is not None:
        return max(1, min(window, n_points))
    return max(MIN_SMOOTH_WINDOW, int(n_points * DEFAULT_SMOOTH_WINDOW_FRAC))


def _smooth_series(
    values: pd.Series,
    *,
    method: str,
    window: int | None,
) -> pd.Series:
    n_points = len(values)
    if method == "none" or n_points < 2:
        return values

    smooth_window = _auto_smooth_window(n_points, window)
    if method == "ema":
        alpha = 2.0 / (smooth_window + 1)
        return values.ewm(alpha=alpha, adjust=False).mean()
    if method == "rolling":
        return values.rolling(window=smooth_window, min_periods=1).mean()
    raise ValueError(f"unknown smoothing method: {method}")


def smooth_learning_curves(
    df: pd.DataFrame,
    *,
    method: str,
    window: int | None,
    value_col: str,
) -> pd.DataFrame:
    """Smooth each run's learning curve before cross-seed aggregation."""
    if method == "none":
        return df

    out = df.sort_values([*RUN_GROUP_COLS, "dataset_size"]).copy()
    out[value_col] = out.groupby(RUN_GROUP_COLS, sort=False)[value_col].transform(
        lambda series: _smooth_series(series, method=method, window=window)
    )
    return out


def _alpha_beta_tag(alpha: float, beta: float) -> str:
    a = str(alpha).replace(".", "p")
    b = str(beta).replace(".", "p")
    return f"a{a}_b{b}"


def _variant_from_config(cfg: dict) -> str:
    if "model" in cfg and "alpha" not in cfg:
        label = cfg.get("label")
        base = cfg["model"]
        return label or base
    alpha_beta = _alpha_beta_tag(cfg["alpha"], cfg["beta"])
    policy = EXPLORE_POLICY.get(alpha_beta)
    if policy is None:
        raise ValueError(f"unknown alpha/beta in config: {cfg!r}")
    base = f"{policy}/{cfg['mode']}/{cfg['bonus']}"
    label = cfg.get("label")
    return label or base


def _task_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    )


def _entries_from_task_dir(
    task_dir: Path,
) -> list[tuple[Path, dict[str, str | int]]]:
    cfg = json.loads((task_dir / "config.json").read_text())
    meta = {
        "env": cfg["env"],
        "variant": _variant_from_config(cfg),
    }
    return [
        (
            run_dir,
            {**meta, "seed": int(run_dir.name.removeprefix("seed_"))},
        )
        for run_dir in _seed_dirs(task_dir)
    ]


def _seed_dirs(parent: Path) -> list[Path]:
    return sorted(
        path
        for path in parent.iterdir()
        if path.is_dir() and path.name.startswith("seed_")
    )


def discover_run_entries(path: Path) -> list[tuple[Path, dict[str, str | int]]]:
    path = path.resolve()
    if _seed_dirs(path) and (path / "config.json").is_file():
        return _entries_from_task_dir(path)

    entries: list[tuple[Path, dict[str, str | int]]] = []
    for task_dir in _task_dirs(path):
        entries.extend(_entries_from_task_dir(task_dir))
    if not entries:
        raise ValueError(f"no task runs found under {path}")
    return entries


def _latest_events_file(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("events.out.tfevents*"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_run_scalars(
    run_dir: Path,
    meta: dict[str, str | int],
    tags: tuple[str, ...],
) -> tuple[list[dict[str, str | int | float]], list[str]]:
    rel = run_dir

    events_file = _latest_events_file(run_dir)
    if events_file is None:
        return [], [f"{rel}: missing events file"]

    ea = EventAccumulator(str(events_file))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    rows: list[dict[str, str | int | float]] = []
    skipped: list[str] = []
    for tag in tags:
        if tag not in available:
            skipped.append(f"{rel}: missing {tag}")
            continue
        rows.extend(
            {
                **meta,
                "tag": tag,
                "dataset_size": int(scalar.step),
                "value": float(scalar.value),
            }
            for scalar in ea.Scalars(tag)
        )
    return rows, skipped


def _load_run_worker(
    args: tuple[str, dict[str, str | int], tuple[str, ...]],
) -> tuple[list[dict[str, str | int | float]], list[str]]:
    run_dir_str, meta, tags = args
    return _load_run_scalars(Path(run_dir_str), meta, tags)


def load_all_scalar_data(
    roots: list[Path],
    workers: int,
    *,
    tags: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, str | int | float]] = []
    skipped: list[str] = []
    entries: list[tuple[Path, dict[str, str | int]]] = []
    for path in roots:
        try:
            entries.extend(discover_run_entries(path))
        except ValueError as exc:
            skipped.append(f"{path}: {exc}")

    worker_args = [(str(run_dir), meta, tags) for run_dir, meta in entries]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for run_rows, skip_reasons in executor.map(_load_run_worker, worker_args):
            skipped.extend(skip_reasons)
            rows.extend(run_rows)

    if not rows:
        root_list = ", ".join(str(path) for path in roots)
        raise RuntimeError(f"no scalar data found under [{root_list}] for tags {tags}")

    return pd.DataFrame(rows), skipped


def split_metric_df(merged: pd.DataFrame, tag: str) -> pd.DataFrame:
    load_col = _load_col_for_tag(tag)
    df = merged.loc[merged["tag"] == tag].copy()
    return (
        df.drop(columns="tag")
        .rename(columns={"value": load_col})
        .reset_index(drop=True)
    )


def _prepare_plot_df(
    df: pd.DataFrame,
    *,
    plot_cfg: PlotConfig,
    smooth: str,
    smooth_window: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    value_col = plot_cfg["value_col"]
    assert isinstance(value_col, str)
    transform = plot_cfg.get("transform")
    if transform is not None:
        assert callable(transform)
        source_col = _load_col_for_cfg(plot_cfg)
        df = df.copy()
        df[value_col] = transform(df[source_col])

    df = smooth_learning_curves(
        df, method=smooth, window=smooth_window, value_col=value_col
    )
    envs_present = [env for env in ENVS if env in df["env"].unique()]
    envs_present.extend(sorted(set(df["env"].unique()) - set(ENVS)))
    variants_present = [v for v in VARIANT_ORDER if v in df["variant"].unique()]
    extra_variants = sorted(set(df["variant"].unique()) - set(VARIANT_ORDER))
    variants_present.extend(extra_variants)
    df = df.copy()
    df["env"] = pd.Categorical(df["env"], categories=envs_present, ordered=True)
    df["variant"] = pd.Categorical(
        df["variant"], categories=variants_present, ordered=True
    )
    return df, envs_present


def _style_axis(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=1,
        bottom=True,
        left=True,
    )


def plot_learning_curves(
    metric_dfs: dict[str, pd.DataFrame],
    metric_ylabels: dict[str, str],
    output: Path,
    dpi: int,
    *,
    smooth: str,
    smooth_window: int | None,
    n_boot: int = 1000,
) -> None:
    prepared: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for tag, df in metric_dfs.items():
        prepared[tag] = _prepare_plot_df(
            df,
            plot_cfg=PLOTS[tag],
            smooth=smooth,
            smooth_window=smooth_window,
        )

    envs_present = next(envs for _, envs in prepared.values())
    metric_tags = [tag for tag in metric_dfs if tag in prepared]
    n_metrics = len(metric_tags)
    n_envs = len(envs_present)
    panel_width = PANEL_HEIGHT * PANEL_ASPECT

    sns.set_theme(style="white")
    fig = plt.figure(
        figsize=(
            n_envs * panel_width,
            n_metrics * PANEL_HEIGHT + LEGEND_ROW_HEIGHT,
        )
    )
    gs = fig.add_gridspec(
        n_metrics + 1,
        n_envs,
        height_ratios=[1] * n_metrics + [LEGEND_ROW_HEIGHT],
        hspace=ROW_PAD,
        wspace=COL_PAD,
        top=0.97,
        bottom=0.06,
        left=0.07,
        right=0.98,
    )

    legend_handles = None
    legend_labels = None
    first_df, _ = next(iter(prepared.values()))
    hue_order = list(first_df["variant"].cat.categories)

    for metric_idx, tag in enumerate(metric_tags):
        df, _ = prepared[tag]
        plot_cfg = PLOTS[tag]
        value_col = plot_cfg["value_col"]
        assert isinstance(value_col, str)
        ylabel = metric_ylabels[tag]
        is_bottom_row = metric_idx == n_metrics - 1

        for env_idx, env in enumerate(envs_present):
            ax = fig.add_subplot(gs[metric_idx, env_idx])
            env_df = df[df["env"] == env]
            sns.lineplot(
                data=env_df,
                x="dataset_size",
                y=value_col,
                hue="variant",
                hue_order=hue_order,
                errorbar=("ci", 95),
                n_boot=n_boot,
                ax=ax,
                legend=env_idx == 0 and metric_idx == 0,
            )
            yscale = plot_cfg.get("yscale")
            if yscale is not None:
                ax.set_yscale(str(yscale))
            ylim_bottom = plot_cfg.get("ylim_bottom")
            if ylim_bottom is not None:
                _, ymax = ax.get_ylim()
                ax.set_ylim(bottom=float(ylim_bottom), top=ymax)
            _style_axis(ax)

            if metric_idx == 0:
                ax.set_title(env, pad=TITLE_PAD)
            if env_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
            if is_bottom_row:
                ax.set_xlabel("Time step")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if env_idx == 0 and metric_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

    if legend_handles is not None and legend_labels is not None:
        legend_ax = fig.add_subplot(gs[n_metrics, :])
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

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"Saved plot to {output}")


def print_summary(
    df: pd.DataFrame,
    skipped: list[str],
    *,
    tag: str,
    value_col: str,
) -> None:
    counts = (
        df.groupby(["env", "variant"], observed=True)
        .agg(runs=("seed", "nunique"), points=(value_col, "count"))
        .reset_index()
    )
    print(f"\nLoaded {tag} data:")
    print(counts.to_string(index=False))

    if skipped:
        print(f"\nSkipped {len(skipped)} run(s):")
        for line in skipped:
            print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learning curves from TensorBoard logs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        nargs="+",
        default=[Path("runs/oracle_eig")],
        metavar="ROOT",
        help="Experiment root(s) containing task_*/config.json run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output PNG path (default: first --root/{DEFAULT_COMBINED_OUTPUT})",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(PLOTS),
        default=("eval/mean_return", "validation/dynamics_mae", "validation/mean_uncertainty"),
        metavar="METRIC",
        help="TensorBoard scalar tags to plot (default: return and dynamics MAE)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel workers for reading TensorBoard logs",
    )
    parser.add_argument(
        "--smooth",
        choices=("none", "ema", "rolling"),
        default="none",
        help=(
            "Per-run smoothing before aggregating across seeds "
            "(default: ema; causal, no centered windows)"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Smoothing window in eval points per run; "
            f"default uses {DEFAULT_SMOOTH_WINDOW_FRAC:.0%} of each curve "
            f"(minimum {MIN_SMOOTH_WINDOW})"
        ),
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        metavar="N",
        help=(
            "Bootstrap resamples for 95%% CI (default 1000; use ~200 for fast preview)"
        ),
    )
    args = parser.parse_args()

    roots = [path.resolve() for path in args.root]
    first_path = roots[0]
    output = args.output or (first_path / DEFAULT_COMBINED_OUTPUT)
    merged_df, skipped = load_all_scalar_data(
        roots, workers=args.workers, tags=tuple(args.metrics)
    )

    metric_dfs: dict[str, pd.DataFrame] = {}
    metric_ylabels: dict[str, str] = {}
    for tag in args.metrics:
        plot_cfg = PLOTS[tag]
        load_col = _load_col_for_tag(tag)
        df = split_metric_df(merged_df, tag)
        if df.empty:
            root_list = ", ".join(str(spec) for spec in args.root)
            print(f"warning: no {tag} data found under [{root_list}]; skipping")
            continue
        print_summary(df, skipped, tag=tag, value_col=load_col)
        metric_dfs[tag] = df
        metric_ylabels[tag] = plot_cfg["ylabel"]

    if not metric_dfs:
        raise RuntimeError(
            f"none of the requested metrics {tuple(args.metrics)} were found "
            f"under {', '.join(args.root)}"
        )

    plot_learning_curves(
        metric_dfs,
        metric_ylabels,
        output,
        args.dpi,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
        n_boot=args.n_boot,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
