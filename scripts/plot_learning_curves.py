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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Keep bootstrap on CPU — TB load uses ProcessPoolExecutor/fork, and a GPU
# bootstrap here previously OOMed allocating a huge cudnn workspace.
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

# Panel titles; wrap long gym ids so they fit at large font sizes.
ENV_DISPLAY_TITLES: dict[str, str] = {
    "MountainCarContinuous-v0": "MountainCar\nContinuous-v0",
}

# Gym classic-control reward thresholds (solved-episode return).
REFERENCE_RETURNS: dict[str, float] = {
    "Acrobot-v1": -100.0,
    "CartPole-v0": 195.0,
    "CartPole-v1": 475.0,
    "MountainCar-v0": -110.0,
    "MountainCarContinuous-v0": 90.0,
    "Pendulum-v1": -200.0,
}
REFERENCE_RETURN_LABEL = "Threshold"

# Sparse return ticks with one tick near the solved threshold.
RETURN_YTICKS: dict[str, list[float]] = {
    "Pendulum-v1": [-1200.0, -700.0, -200.0],
    "MountainCar-v0": [-200.0, -150.0, -100.0],
    "MountainCarContinuous-v0": [0.0, 100.0],
    "CartPole-v1": [0.0, 250.0, 500.0],
    "Acrobot-v1": [-500.0, -300.0, -100.0],
}

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

PlotConfig = dict[str, str | bool | int | float | Callable[[pd.Series], pd.Series]]

PLOTS: dict[str, PlotConfig] = {
    "eval/mean_return": {
        "value_col": "mean_return",
        "ylabel": "Return",
        "reference_returns": True,
        "ylim_pad": 0.06,
    },
    "validation/dynamics_mae": {
        "value_col": "dynamics_mae",
        "ylabel": "Model\nError",
        "yscale": "log",
        "ylim_pad": 0.08,
        "min_log_yticks": 2,
        "max_log_yticks": 3,
        "hide_minor_yticks": True,
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
        # Log-scale CI ribbons often sit flush against the axes; pad in log
        # space from the actual ribbon extents and drop minor ticks so labels
        # don't crowd/overlap the curves.
        "ylim_pad": 0.22,
        "min_log_yticks": 2,
        "max_log_yticks": 3,
        "hide_minor_yticks": True,
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
PANEL_HEIGHT = 3.8
PANEL_ASPECT = 1.55
LEGEND_ROW_HEIGHT = 0.7
TITLE_PAD = 8
SAVE_PAD_INCHES = 0.3
LINE_ALPHA = 0.75


def _auto_smooth_window(n_points: int, window: int | None) -> int:
    if window is not None:
        return max(1, min(window, n_points))
    return max(MIN_SMOOTH_WINDOW, int(n_points * DEFAULT_SMOOTH_WINDOW_FRAC))


def _smooth_series(
    values: pd.Series,
    *,
    method: str,
    window: int | None,
    ema_alpha: float | None = None,
    ema_adjust: bool = False,
) -> pd.Series:
    n_points = len(values)
    if method == "none" or n_points < 2:
        return values

    smooth_window = _auto_smooth_window(n_points, window)
    if method == "ema":
        alpha = (
            float(ema_alpha)
            if ema_alpha is not None
            else 2.0 / (smooth_window + 1)
        )
        return values.ewm(alpha=alpha, adjust=ema_adjust).mean()
    if method == "rolling":
        return values.rolling(window=smooth_window, min_periods=1).mean()
    raise ValueError(f"unknown smoothing method: {method}")


def smooth_learning_curves(
    df: pd.DataFrame,
    *,
    method: str,
    window: int | None,
    value_col: str,
    group_cols: list[str] | None = None,
    ema_alpha: float | None = None,
    ema_adjust: bool = False,
) -> pd.DataFrame:
    """Smooth each group's curve (default: per-run) along ``dataset_size``."""
    if method == "none":
        return df

    groups = group_cols if group_cols is not None else list(RUN_GROUP_COLS)
    out = df.sort_values([*groups, "dataset_size"]).copy()
    out[value_col] = out.groupby(groups, sort=False)[value_col].transform(
        lambda series: _smooth_series(
            series,
            method=method,
            window=window,
            ema_alpha=ema_alpha,
            ema_adjust=ema_adjust,
        )
    )
    return out


def _bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)`` for a 1-d sample across seeds."""
    global _jax_rng
    vals = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(vals))
    if vals.size < 2 or n_boot <= 0:
        return mean, mean, mean
    _jax_rng, key = jax.random.split(_jax_rng)
    boots = np.asarray(
        _jax_mean_boot_fn(n_boot)(jnp.asarray(vals, dtype=jnp.float32), key)
    )
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(lo), float(hi)


def aggregate_seed_curves(
    df: pd.DataFrame,
    *,
    value_col: str,
    n_boot: int,
) -> pd.DataFrame:
    """Mean ± bootstrap CI across seeds at each ``dataset_size`` (no smoothing)."""
    rows: list[dict[str, object]] = []
    for (env, variant), group in df.groupby(["env", "variant"], observed=True, sort=False):
        for step, step_group in group.groupby("dataset_size", sort=True):
            mean, lo, hi = _bootstrap_mean_ci(
                step_group[value_col].to_numpy(), n_boot=n_boot
            )
            rows.append(
                {
                    "env": env,
                    "variant": variant,
                    "dataset_size": int(step),
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def smooth_aggregate_curves(
    agg: pd.DataFrame,
    *,
    method: str,
    window: int | None,
    ema_alpha: float | None = None,
    ema_adjust: bool = False,
) -> pd.DataFrame:
    """Smooth mean and CI bands after cross-seed aggregation."""
    if method == "none" or agg.empty:
        return agg
    out = agg.sort_values(["env", "variant", "dataset_size"]).copy()
    for col in ("mean", "ci_lo", "ci_hi"):
        out[col] = out.groupby(["env", "variant"], sort=False)[col].transform(
            lambda series: _smooth_series(
                series,
                method=method,
                window=window,
                ema_alpha=ema_alpha,
                ema_adjust=ema_adjust,
            )
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
    prefix_root = len(roots) > 1
    for path in roots:
        try:
            path_entries = discover_run_entries(path)
        except ValueError as exc:
            skipped.append(f"{path}: {exc}")
            continue
        if prefix_root:
            root_name = path.name
            path_entries = [
                (run_dir, {**meta, "variant": f"{root_name}/{meta['variant']}"})
                for run_dir, meta in path_entries
            ]
        entries.extend(path_entries)

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


def filter_variants(
    df: pd.DataFrame, variants: list[str] | None
) -> pd.DataFrame:
    if not variants:
        return df
    present = set(df["variant"].unique())
    missing = [v for v in variants if v not in present]
    if missing:
        print(
            "warning: requested variants not found: "
            + ", ".join(missing)
        )
    keep = [v for v in variants if v in present]
    if not keep:
        return df.iloc[0:0].copy()
    return df[df["variant"].isin(keep)].copy()


def _prepare_plot_df(
    df: pd.DataFrame,
    *,
    plot_cfg: PlotConfig,
    smooth: str,
    smooth_window: int | None,
    variant_order: list[str] | None = None,
    ema_alpha: float | None = None,
    ema_adjust: bool = False,
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
        df,
        method=smooth,
        window=smooth_window,
        value_col=value_col,
        ema_alpha=ema_alpha,
        ema_adjust=ema_adjust,
    )
    envs_present = [env for env in ENVS if env in df["env"].unique()]
    envs_present.extend(sorted(set(df["env"].unique()) - set(ENVS)))
    if variant_order:
        variants_present = [v for v in variant_order if v in df["variant"].unique()]
        extras = sorted(set(df["variant"].unique()) - set(variants_present))
        variants_present.extend(extras)
    else:
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


def _pad_ylim(ax: plt.Axes, *, yscale: str | None, pad: float) -> None:
    """Expand y-limits so CI ribbons / reference lines are not clipped."""
    if pad <= 0:
        return
    ymin, ymax = _artist_ylim(ax)
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return
    if yscale == "log":
        if ymin <= 0 or ymax <= 0:
            return
        log_min, log_max = np.log10(ymin), np.log10(ymax)
        span = log_max - log_min
        if span <= 0:
            span = 1.0
        ax.set_ylim(10 ** (log_min - pad * span), 10 ** (log_max + pad * span))
        return
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    ax.set_ylim(ymin - pad * span, ymax + pad * span)


def _artist_ylim(ax: plt.Axes) -> tuple[float, float]:
    """Y extent of lines + fill_between collections (CI ribbons)."""
    ys: list[np.ndarray] = []
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float).ravel()
        y = y[np.isfinite(y)]
        if y.size:
            ys.append(y)
    for coll in ax.collections:
        for path in coll.get_paths():
            y = np.asarray(path.vertices[:, 1], dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                ys.append(y)
    if not ys:
        return np.nan, np.nan
    all_y = np.concatenate(ys)
    return float(np.min(all_y)), float(np.max(all_y))


def _ensure_y_includes(ax: plt.Axes, value: float, *, yscale: str | None) -> None:
    ymin, ymax = ax.get_ylim()
    if yscale == "log":
        if value <= 0:
            return
        ax.set_ylim(min(ymin, value), max(ymax, value))
        return
    ax.set_ylim(min(ymin, value), max(ymax, value))


def _apply_return_yticks(ax: plt.Axes, env: str) -> None:
    """Set return tick locations; expand ylim only if needed so ticks are visible."""
    ticks = RETURN_YTICKS.get(env)
    if not ticks:
        return
    ymin, ymax = ax.get_ylim()
    lo, hi = min(ticks), max(ticks)
    # Expand to include requested ticks, never shrink (would clip data).
    ax.set_ylim(min(ymin, lo), max(ymax, hi))
    ax.yaxis.set_major_locator(FixedLocator(ticks))


_SUPER_TRANS = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _one_sig_digit(x: float) -> float:
    """Round a positive value to one significant digit."""
    if not np.isfinite(x) or x <= 0:
        return float(x)
    exp = int(np.floor(np.log10(x)))
    return float(round(x / 10.0**exp) * 10.0**exp)


def _format_one_sig(x: float, _pos: int | None = None) -> str:
    """Format log ticks as pure decades ``10ⁿ`` (no mantissa / 0 sig figs)."""
    if not np.isfinite(x) or x <= 0:
        return ""
    exp = int(round(np.log10(x)))
    return f"10{str(exp).translate(_SUPER_TRANS)}"


def _ensure_min_log_yticks(
    ax: plt.Axes, min_ticks: int = 2, max_ticks: int = 3
) -> None:
    """Place 2–3 decade log ticks inside the current ylim (ylim unchanged)."""
    if min_ticks < 2:
        return
    max_ticks = max(max_ticks, min_ticks)
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin <= 0 or ymax <= ymin:
        return

    lo = int(np.floor(np.log10(ymin)))
    hi = int(np.ceil(np.log10(ymax)))
    if hi < lo:
        hi = lo
    # Decades strictly inside the view when possible.
    candidates = [e for e in range(lo, hi + 1) if ymin <= 10.0**e <= ymax]
    if len(candidates) < min_ticks:
        candidates = list(range(lo, hi + 1))
    if len(candidates) > max_ticks:
        idx = np.linspace(0, len(candidates) - 1, max_ticks)
        candidates = [candidates[int(round(i))] for i in idx]
        # Deduplicate while preserving order.
        dedup: list[int] = []
        for e in candidates:
            if not dedup or e != dedup[-1]:
                dedup.append(e)
        candidates = dedup
    ticks = [10.0**e for e in candidates]
    if len(ticks) < min_ticks:
        ticks = [10.0**lo, 10.0**hi]

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_one_sig))
    ax.yaxis.set_minor_locator(NullLocator())


def plot_learning_curves(
    metric_dfs: dict[str, pd.DataFrame],
    metric_ylabels: dict[str, str],
    output: Path,
    dpi: int,
    *,
    smooth: str,
    smooth_window: int | None,
    n_boot: int = 1000,
    line_alpha: float = LINE_ALPHA,
    variant_order: list[str] | None = None,
    font_size: int | None = None,
    serif: bool = False,
    scale: float = 1.0,
    ema_alpha: float | None = None,
    ema_adjust: bool = False,
) -> None:
    prepared: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for tag, df in metric_dfs.items():
        prepared[tag] = _prepare_plot_df(
            df,
            plot_cfg=PLOTS[tag],
            smooth=smooth,
            smooth_window=smooth_window,
            variant_order=variant_order,
            ema_alpha=ema_alpha,
            ema_adjust=ema_adjust,
        )

    envs_present = next(envs for _, envs in prepared.values())
    metric_tags = [tag for tag in metric_dfs if tag in prepared]
    n_metrics = len(metric_tags)
    n_envs = len(envs_present)
    panel_height = PANEL_HEIGHT * scale
    panel_width = panel_height * PANEL_ASPECT
    legend_row_height = LEGEND_ROW_HEIGHT * scale

    rc: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if font_size is not None:
        # Seaborn's default context pins axes/tick/legend sizes (~11–12pt).
        # Setting only font.size leaves those fixed — mathtext y-ticks then
        # look large while titles/labels/legend stay small. Override all.
        # Ticks stay smaller than titles/labels so panels don't crowd.
        tick_size = max(6, int(round(font_size * 0.75)))
        rc.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": tick_size,
                "ytick.labelsize": tick_size,
                "legend.fontsize": font_size,
                "legend.title_fontsize": font_size,
            }
        )
    if serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = [
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
        ]
    sns.set_theme(style="white", rc=rc)
    # Re-apply after set_theme in case context merge dropped keys.
    if font_size is not None:
        tick_size = max(6, int(round(font_size * 0.75)))
        plt.rcParams.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": tick_size,
                "ytick.labelsize": tick_size,
                "legend.fontsize": font_size,
                "legend.title_fontsize": font_size,
            }
        )
    fig = plt.figure(
        figsize=(
            n_envs * panel_width,
            n_metrics * panel_height + legend_row_height,
        ),
    )
    # Extra horizontal gap so per-env y-tick labels don't collide.
    gs = fig.add_gridspec(
        n_metrics,
        n_envs,
        wspace=0.45,
        hspace=0.28,
    )

    legend_handles = None
    legend_labels = None
    first_df, _ = next(iter(prepared.values()))
    hue_order = list(first_df["variant"].cat.categories)
    reference_handle = None

    for metric_idx, tag in enumerate(metric_tags):
        df, _ = prepared[tag]
        plot_cfg = PLOTS[tag]
        value_col = plot_cfg["value_col"]
        assert isinstance(value_col, str)
        ylabel = metric_ylabels[tag]
        is_bottom_row = metric_idx == n_metrics - 1
        yscale = plot_cfg.get("yscale")
        yscale_str = str(yscale) if yscale is not None else None

        for env_idx, env in enumerate(envs_present):
            ax = fig.add_subplot(gs[metric_idx, env_idx])
            env_df = df[df["env"] == env]
            # n_boot<=0 skips CI for fast layout previews.
            errorbar = None if n_boot <= 0 else ("ci", 95)
            sns.lineplot(
                data=env_df,
                x="dataset_size",
                y=value_col,
                hue="variant",
                hue_order=hue_order,
                errorbar=errorbar,
                n_boot=max(n_boot, 1),
                ax=ax,
                legend=env_idx == 0 and metric_idx == 0,
            )
            for line in ax.lines:
                line.set_alpha(line_alpha)
                line.set_zorder(3)
            for coll in ax.collections:
                coll.set_zorder(2)
            if yscale is not None:
                ax.set_yscale(str(yscale))
            ylim_bottom = plot_cfg.get("ylim_bottom")
            if ylim_bottom is not None:
                _, ymax = ax.get_ylim()
                ax.set_ylim(bottom=float(ylim_bottom), top=ymax)

            if env_idx == 0 and metric_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

            if plot_cfg.get("reference_returns"):
                ref = REFERENCE_RETURNS.get(str(env))
                if ref is not None:
                    _ensure_y_includes(ax, ref, yscale=yscale_str)
                    ax.axhline(
                        ref,
                        color="0.35",
                        linestyle="--",
                        linewidth=1.2,
                        zorder=0.5,
                    )
                    if reference_handle is None:
                        reference_handle = Line2D(
                            [0],
                            [0],
                            color="0.35",
                            linestyle="--",
                            linewidth=1.2,
                            label=REFERENCE_RETURN_LABEL,
                        )

            ylim_pad = plot_cfg.get("ylim_pad")
            if ylim_pad is not None:
                _pad_ylim(ax, yscale=yscale_str, pad=float(ylim_pad))

            if plot_cfg.get("reference_returns"):
                _apply_return_yticks(ax, str(env))
            else:
                log_numticks = plot_cfg.get("log_numticks")
                if yscale_str == "log" and log_numticks is not None:
                    ax.yaxis.set_major_locator(
                        LogLocator(base=10.0, numticks=int(log_numticks))
                    )
                min_log_yticks = plot_cfg.get("min_log_yticks")
                if yscale_str == "log" and min_log_yticks is not None:
                    max_log_yticks = plot_cfg.get("max_log_yticks", 3)
                    _ensure_min_log_yticks(
                        ax,
                        min_ticks=int(min_log_yticks),
                        max_ticks=int(max_log_yticks),
                    )
                if plot_cfg.get("hide_minor_yticks"):
                    ax.yaxis.set_minor_locator(NullLocator())

            _style_axis(ax)
            # Fixed x ticks at the episode/time endpoints used in these runs.
            ax.set_xlim(0, 1000)
            ax.xaxis.set_major_locator(FixedLocator([0, 1000]))

            if metric_idx == 0:
                ax.set_title(
                    ENV_DISPLAY_TITLES.get(str(env), str(env)),
                    pad=TITLE_PAD,
                )
            if env_idx == 0:
                # Horizontal, multi-line y-labels (e.g. "Model\\nError").
                ax.set_ylabel(
                    ylabel,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                )
            else:
                ax.set_ylabel("")
            if is_bottom_row:
                ax.set_xlabel("Time step")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    if legend_handles is not None and legend_labels is not None:
        # Keep the threshold line on the return panels, but omit it from the legend.
        pairs = [
            (h, lab)
            for h, lab in zip(legend_handles, legend_labels)
            if lab != REFERENCE_RETURN_LABEL
        ]
        legend_handles = [h for h, _ in pairs]
        legend_labels = [lab for _, lab in pairs]
        # Reserve space under x-labels, then place the legend clearly below.
        fig.tight_layout(rect=(0, 0.22, 1, 1), pad=0.5, w_pad=1.6, h_pad=0.7)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_labels),
            frameon=False,
            columnspacing=1.4,
            handlelength=1.8,
            borderaxespad=0.0,
        )
    else:
        fig.tight_layout(pad=0.5, w_pad=1.6, h_pad=0.7)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
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
        "--smooth-alpha",
        type=float,
        default=None,
        metavar="A",
        help=(
            "EMA alpha (overrides --smooth-window). Typical values: 0.05, 0.1."
        ),
    )
    parser.add_argument(
        "--smooth-adjust",
        action="store_true",
        help="Use pandas ewm(adjust=True) for EMA (default: adjust=False).",
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
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=LINE_ALPHA,
        metavar="A",
        help=f"Opacity of mean learning-curve lines (default {LINE_ALPHA})",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        metavar="PT",
        help="Base font size in points for all text (default: matplotlib default)",
    )
    parser.add_argument(
        "--serif",
        action="store_true",
        help="Use a Times New Roman-compatible serif font instead of the default sans-serif",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="F",
        help=(
            "Shrink/grow the panel grid canvas by this factor (font size is "
            "absolute, not scaled) — set close to the final embed width/16.1in "
            "so --font-size renders WYSIWYG instead of getting shrunk further "
            "by the LaTeX \\includegraphics scale-to-fit"
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        metavar="VARIANT",
        help=(
            "Only plot these variant labels (order preserved). "
            "With multiple --root, labels are rooted as '<root>/<variant>'."
        ),
    )
    parser.add_argument(
        "--rename-variants",
        nargs="+",
        default=None,
        metavar="OLD=NEW",
        help=(
            "Rewrite variant labels for display, e.g. "
            "explore_sample_eig=M-PPO+TS+EIG. Applied after --variants "
            "filtering; --variants order is preserved under the new names."
        ),
    )
    args = parser.parse_args()

    roots = [path.resolve() for path in args.root]
    first_path = roots[0]
    output = args.output or (first_path / DEFAULT_COMBINED_OUTPUT)
    merged_df, skipped = load_all_scalar_data(
        roots, workers=args.workers, tags=tuple(args.metrics)
    )
    if args.variants:
        merged_df = filter_variants(merged_df, args.variants)
        if merged_df.empty:
            raise RuntimeError(
                f"none of the requested variants {tuple(args.variants)} were found "
                f"under {', '.join(args.root)}"
            )

    variant_order = args.variants
    if args.rename_variants:
        rename_map = dict(pair.split("=", 1) for pair in args.rename_variants)
        merged_df["variant"] = merged_df["variant"].map(
            lambda v: rename_map.get(v, v)
        )
        if variant_order:
            variant_order = [rename_map.get(v, v) for v in variant_order]

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
        line_alpha=args.line_alpha,
        variant_order=variant_order,
        font_size=args.font_size,
        serif=args.serif,
        scale=args.scale,
        ema_alpha=args.smooth_alpha,
        ema_adjust=args.smooth_adjust,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
