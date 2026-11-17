#!/usr/bin/env python3
"""State-space trajectories of the real-environment data-collecting policy.

Companion to ``G1_mue_vs_true_reward.pdf``: same experiment
(``classic_plan_every_cheap_model_long_variants``), same two variants
(explore_sample_eig vs exploit_sample_eig). Each real step collected by the
behavior policy is drawn in the env's 2-D configuration space, with the line
colored by training timestep so later visits sit on top of earlier ones.

Data: TensorBoard ``dataset/obs_*`` (and terminated/truncated to break the
polyline at episode boundaries). ``steps_per_rollout=1``, so one logged
transition per outer iteration.

Usage:
    uv run python scripts/plot_exploration_trajectories.py \
      --root runs/classic_plan_every_cheap_model_long_variants \
      --variants explore_sample_eig exploit_sample_eig \
      --rename-variants \
        explore_sample_eig="EIG Intrinsic Reward" \
        exploit_sample_eig="Extrinsic Reward" \
      --font-size 10 \
      --also-tmax 100 \
      --output runs/compare_plan_every/G1_exploration_trajectories.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from tensorboard.backend.event_processing import event_file_loader

ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
]

ENV_DISPLAY_TITLES: dict[str, str] = {
    "MountainCarContinuous-v0": "MountainCar\nContinuous-v0",
}

SAVE_PAD_INCHES = 0.25
CMAP = "viridis"
LINEWIDTH = 0.65
LINE_ALPHA = 0.45
PANEL_INCHES = 3.2

_OBS_TAGS = {
    "Pendulum-v1": ("dataset/obs_0", "dataset/obs_1", "dataset/obs_2"),
    "MountainCar-v0": ("dataset/obs_0", "dataset/obs_1"),
    "MountainCarContinuous-v0": ("dataset/obs_0", "dataset/obs_1"),
    "CartPole-v1": ("dataset/obs_0", "dataset/obs_2"),
    "Acrobot-v1": (
        "dataset/obs_0",
        "dataset/obs_1",
        "dataset/obs_2",
        "dataset/obs_3",
    ),
}
_BREAK_TAGS = ("dataset/terminated", "dataset/truncated")

AXES_LABELS = {
    "Pendulum-v1": (r"$\theta$", r"$\dot{\theta}$"),
    "MountainCar-v0": ("Position", "Velocity"),
    "MountainCarContinuous-v0": ("Position", "Velocity"),
    "CartPole-v1": (r"$x$", r"$\theta$"),
    "Acrobot-v1": (r"$\theta_1$", r"$\theta_2$"),
}

DEFAULT_LIMS = {
    "Pendulum-v1": ((-np.pi, np.pi), (-8.0, 8.0)),
    "MountainCar-v0": ((-1.2, 0.6), (-0.07, 0.07)),
    "MountainCarContinuous-v0": ((-1.2, 0.6), (-0.07, 0.07)),
    "CartPole-v1": ((-2.4, 2.4), (-0.25, 0.25)),
    "Acrobot-v1": ((-np.pi, np.pi), (-np.pi, np.pi)),
}

GOAL_X = {
    "MountainCar-v0": 0.5,
    "MountainCarContinuous-v0": 0.45,
}

WRAP_XY = {
    "Pendulum-v1": (True, False),
    "Acrobot-v1": (True, True),
}

ANGLE_X = {"Pendulum-v1", "Acrobot-v1"}
ANGLE_Y = {"Acrobot-v1"}


def _extract_value(v) -> float:
    if v.HasField("simple_value"):
        return v.simple_value
    if v.HasField("tensor"):
        return float(v.tensor.float_val[0])
    raise ValueError(f"unrecognized value proto for tag {v.tag!r}")


def _tfevents_files(seed_dir: str) -> list[str]:
    files = glob.glob(os.path.join(seed_dir, "events.out.tfevents.*"))
    return sorted(files, key=lambda p: float(os.path.basename(p).split(".")[3]))


def load_tags(seed_dir: str, tags: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load scalar tags from one seed dir, later tfevents overriding earlier."""
    wanted = set(tags)
    step_to_val: dict[str, dict[int, float]] = {t: {} for t in tags}
    for path in _tfevents_files(seed_dir):
        loader = event_file_loader.EventFileLoader(path)
        for event in loader.Load():
            if not event.summary.value:
                continue
            for v in event.summary.value:
                if v.tag in wanted:
                    step_to_val[v.tag][event.step] = _extract_value(v)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        steps = np.array(sorted(step_to_val[tag]), dtype=np.int64)
        vals = np.array([step_to_val[tag][s] for s in steps], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def project_xy(env: str, series: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if env == "Pendulum-v1":
        return np.arctan2(series["dataset/obs_1"], series["dataset/obs_0"]), series[
            "dataset/obs_2"
        ]
    if env in ("MountainCar-v0", "MountainCarContinuous-v0"):
        return series["dataset/obs_0"], series["dataset/obs_1"]
    if env == "CartPole-v1":
        return series["dataset/obs_0"], series["dataset/obs_2"]
    if env == "Acrobot-v1":
        th1 = np.arctan2(series["dataset/obs_1"], series["dataset/obs_0"])
        th2 = np.arctan2(series["dataset/obs_3"], series["dataset/obs_2"])
        return th1, th2
    raise ValueError(f"no 2-D projection defined for {env!r}")


def _align_series(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]], tags: tuple[str, ...]
) -> dict[str, np.ndarray] | None:
    common = None
    for tag in tags:
        steps, _ = loaded[tag]
        if steps.size == 0:
            return None
        common = set(steps.tolist()) if common is None else common.intersection(steps.tolist())
    if not common:
        return None
    order = np.array(sorted(common), dtype=np.int64)
    aligned: dict[str, np.ndarray] = {"step": order}
    for tag in tags:
        step, val = loaded[tag]
        lookup = dict(zip(step.tolist(), val.tolist(), strict=True))
        aligned[tag] = np.array([lookup[s] for s in order], dtype=np.float64)
    return aligned


def _load_seed_xy(args: tuple[str, str]) -> dict[str, np.ndarray] | None:
    seed_dir, env = args
    tags = _OBS_TAGS[env] + _BREAK_TAGS
    loaded = load_tags(seed_dir, tags)
    aligned = _align_series(loaded, tags)
    if aligned is None:
        return None
    x, y = project_xy(env, aligned)
    episode_break = (aligned["dataset/terminated"] > 0.5) | (
        aligned["dataset/truncated"] > 0.5
    )
    return {"x": x, "y": y, "t": aligned["step"].astype(np.float64), "break": episode_break}


def traj_segments(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    episode_break: np.ndarray,
    wrap_x: bool,
    wrap_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Polyline segments colored at the originating timestep; split at resets/wraps."""
    if x.size < 2:
        return np.empty((0, 2, 2)), np.empty((0,))
    pts = np.column_stack((x, y))
    segs = np.stack((pts[:-1], pts[1:]), axis=1)
    keep = ~episode_break[:-1]
    if wrap_x:
        keep &= np.abs(np.diff(x)) < np.pi
    if wrap_y:
        keep &= np.abs(np.diff(y)) < np.pi
    return segs[keep], t[:-1][keep]


def _format_pi(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-8:
        return r"$0$"
    n = value / np.pi
    if abs(n - 1) < 1e-8:
        return r"$\pi$"
    if abs(n + 1) < 1e-8:
        return r"$-\pi$"
    return rf"${n:g}\pi$"


def _apply_angle_ticks(ax: plt.Axes, axis: str) -> None:
    locator = MultipleLocator(base=np.pi)
    formatter = FuncFormatter(_format_pi)
    if axis in ("x", "both"):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)


def _style_axis(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def _shared_limits(
    env: str, trajs: list[dict[str, np.ndarray]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    (xlo, xhi), (ylo, yhi) = DEFAULT_LIMS[env]
    if trajs:
        xs = np.concatenate([tr["x"] for tr in trajs])
        ys = np.concatenate([tr["y"] for tr in trajs])
        if env not in ANGLE_X:
            xlo, xhi = min(xlo, float(xs.min())), max(xhi, float(xs.max()))
        if env not in ANGLE_Y:
            ylo, yhi = min(ylo, float(ys.min())), max(yhi, float(ys.max()))
    xpad = 0.03 * (xhi - xlo) if env not in ANGLE_X else 0.0
    ypad = 0.04 * (yhi - ylo) if env not in ANGLE_Y else 0.0
    return (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)


def occupancy_fraction(
    trajs: list[dict[str, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bins: int = 20,
) -> float:
    if not trajs:
        return float("nan")
    x = np.concatenate([tr["x"] for tr in trajs])
    y = np.concatenate([tr["y"] for tr in trajs])
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    return float((hist > 0).mean())


def discover_runs(root: Path, variants: list[str]) -> list[tuple[Path, str, str]]:
    wanted = set(variants)
    out: list[tuple[Path, str, str]] = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("task_")):
        cfg_path = task_dir / "config.json"
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text())
        variant = cfg.get("label")
        env = cfg.get("env")
        if variant not in wanted or env not in _OBS_TAGS:
            continue
        for seed_dir in sorted(task_dir.glob("seed_*")):
            out.append((seed_dir, str(env), str(variant)))
    return out


def _apply_rc(font_size: int, serif: bool) -> None:
    rc: dict[str, object] = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }
    if serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
        rc["mathtext.fontset"] = "stix"
    else:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
        rc["mathtext.fontset"] = "dejavusans"
    sns.set_theme(style="white", rc=rc)
    # seaborn can drop font.family / mathtext; re-apply after the theme merge.
    plt.rcParams.update(rc)


def plot_trajectories(
    by_variant: dict[str, dict[str, list[dict[str, np.ndarray]]]],
    variant_order: list[str],
    envs: list[str],
    output: Path,
    *,
    font_size: int,
    serif: bool,
    dpi: int,
    env_lims: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    t_clim: float | None = None,
) -> None:
    _apply_rc(font_size, serif)
    n_rows = len(variant_order)
    n_envs = len(envs)
    n_seeds = max(
        (len(trajs) for env_map in by_variant.values() for trajs in env_map.values()),
        default=1,
    )
    if n_seeds <= 1:
        line_alpha = 0.95
        linewidth = 1.35
    elif n_seeds <= 5:
        line_alpha = LINE_ALPHA
        linewidth = LINEWIDTH
    else:
        line_alpha = 0.32
        linewidth = LINEWIDTH

    fig = plt.figure(
        figsize=(n_envs * PANEL_INCHES + 0.7, n_rows * PANEL_INCHES + 0.2),
        layout="constrained",
    )
    gs = fig.add_gridspec(n_rows, n_envs + 1, width_ratios=[0.16] + [1] * n_envs)
    axes = np.empty((n_rows, n_envs), dtype=object)

    all_t_max = 1000.0
    if t_clim is not None:
        all_t_max = float(t_clim)
    norm = Normalize(vmin=0.0, vmax=all_t_max)

    for row, variant in enumerate(variant_order):
        lab = fig.add_subplot(gs[row, 0])
        lab.axis("off")
        lab.text(
            0.5,
            0.5,
            variant,
            rotation=90,
            va="center",
            ha="center",
            transform=lab.transAxes,
        )
        for col, env in enumerate(envs):
            ax = fig.add_subplot(gs[row, col + 1])
            axes[row, col] = ax
            trajs = by_variant[variant].get(env, [])
            if env_lims is not None:
                xlim, ylim = env_lims[env]
            else:
                both = [tr for v in variant_order for tr in by_variant[v].get(env, [])]
                xlim, ylim = _shared_limits(env, both)
            wrap_x, wrap_y = WRAP_XY.get(env, (False, False))
            segs_list: list[np.ndarray] = []
            t_list: list[np.ndarray] = []
            for tr in trajs:
                segs, t_seg = traj_segments(
                    tr["x"], tr["y"], tr["t"], tr["break"], wrap_x, wrap_y
                )
                if segs.size:
                    segs_list.append(segs)
                    t_list.append(t_seg)
            if segs_list:
                segs = np.concatenate(segs_list, axis=0)
                t_seg = np.concatenate(t_list, axis=0)
                order = np.argsort(t_seg)
                lc = LineCollection(
                    segs[order],
                    cmap=CMAP,
                    norm=norm,
                    array=t_seg[order],
                    linewidths=linewidth,
                    alpha=line_alpha,
                    rasterized=True,
                    zorder=2,
                )
                ax.add_collection(lc)
            goal = GOAL_X.get(env)
            if goal is not None:
                ax.axvline(goal, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if env in ANGLE_X:
                _apply_angle_ticks(ax, "x")
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
            if env in ANGLE_Y:
                _apply_angle_ticks(ax, "y")
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
            _style_axis(ax)
            xlabel, ylabel = AXES_LABELS[env]
            if row == 0:
                ax.set_title(ENV_DISPLAY_TITLES.get(env, env), pad=10)
            if row == n_rows - 1:
                ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(ylabel)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Time step")
    cbar.set_ticks(np.linspace(0.0, all_t_max, 5))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    png_path = output.with_suffix(".png")
    if png_path != output:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
        print(f"Saved plot to {png_path}")
    plt.close(fig)
    print(f"Saved plot to {output}")


def clip_trajectories(
    by_variant: dict[str, dict[str, list[dict[str, np.ndarray]]]],
    tmax: int,
) -> dict[str, dict[str, list[dict[str, np.ndarray]]]]:
    """Keep observations with step <= tmax (inclusive)."""
    out: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {}
    for variant, env_map in by_variant.items():
        out[variant] = {}
        for env, trajs in env_map.items():
            clipped: list[dict[str, np.ndarray]] = []
            for tr in trajs:
                mask = tr["t"] <= tmax
                if not np.any(mask):
                    continue
                clipped.append({key: val[mask] for key, val in tr.items()})
            out[variant][env] = clipped
    return out


def env_axis_limits(
    by_variant: dict[str, dict[str, list[dict[str, np.ndarray]]]],
    variant_order: list[str],
    envs: list[str],
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    lims = {}
    for env in envs:
        both = [tr for v in variant_order for tr in by_variant[v].get(env, [])]
        lims[env] = _shared_limits(env, both)
    return lims


def print_coverage(
    by_variant: dict[str, dict[str, list[dict[str, np.ndarray]]]],
    variant_order: list[str],
    envs: list[str],
    env_lims: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> None:
    print("\nState-space occupancy (20×20 bins over shared axis limits):")
    for env in envs:
        both = [tr for v in variant_order for tr in by_variant[v].get(env, [])]
        xlim, ylim = env_lims[env] if env_lims is not None else _shared_limits(env, both)
        parts = []
        for variant in variant_order:
            trajs = by_variant[variant].get(env, [])
            frac = occupancy_fraction(trajs, xlim, ylim)
            n = len(trajs)
            extra = ""
            if env in GOAL_X and trajs:
                thr = GOAL_X[env]
                n_reach = sum(int(np.any(tr["x"] >= thr)) for tr in trajs)
                extra = f", {n_reach}/{n} seeds reached x≥{thr:g}"
            parts.append(f"{variant}: {frac:.1%}{extra}")
        print(f"  {env}: " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot real-env exploration trajectories colored by timestep."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/classic_plan_every_cheap_model_long_variants"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/compare_plan_every/G1_exploration_trajectories.pdf"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["explore_sample_eig", "exploit_sample_eig"],
    )
    parser.add_argument(
        "--rename-variants",
        nargs="+",
        default=None,
        metavar="OLD=NEW",
    )
    parser.add_argument("--font-size", type=int, default=10)
    parser.add_argument("--serif", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Cap seeds per env/variant (default: all).",
    )
    parser.add_argument(
        "--tmax",
        type=int,
        default=None,
        help="If set, only plot observations with step <= this value.",
    )
    parser.add_argument(
        "--also-tmax",
        type=int,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Also write cropped copies (same axis limits as the full plot) "
            "keeping step <= T, saved as <stem>_tT.pdf next to --output."
        ),
    )
    args = parser.parse_args()

    rename_map = (
        dict(pair.split("=", 1) for pair in args.rename_variants)
        if args.rename_variants
        else {}
    )

    entries = discover_runs(args.root.resolve(), args.variants)
    if not entries:
        raise RuntimeError(f"no matching runs under {args.root}")

    grouped: dict[tuple[str, str], list[Path]] = {}
    for seed_dir, env, variant in entries:
        grouped.setdefault((variant, env), []).append(seed_dir)

    jobs: list[tuple[str, str, str, str]] = []
    for variant in args.variants:
        for env in ENVS:
            dirs = sorted(grouped.get((variant, env), []), key=lambda p: p.name)
            if args.max_seeds is not None:
                dirs = dirs[: args.max_seeds]
            for run_dir in dirs:
                jobs.append((str(run_dir), env, variant, run_dir.name))

    print(f"Loading {len(jobs)} seed runs with {args.workers} workers...")
    worker_args = [(seed_dir, env) for seed_dir, env, _, _ in jobs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        loaded = list(ex.map(_load_seed_xy, worker_args, chunksize=1))

    by_variant: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {
        rename_map.get(v, v): {} for v in args.variants
    }
    n_skip = 0
    for (seed_dir, env, variant, seed_name), traj in zip(jobs, loaded, strict=True):
        display = rename_map.get(variant, variant)
        if traj is None or traj["x"].size == 0:
            n_skip += 1
            print(f"  skip {env} {variant} {seed_name} ({seed_dir})")
            continue
        by_variant[display].setdefault(env, []).append(traj)

    variant_order = [rename_map.get(v, v) for v in args.variants]
    envs_present = [
        env for env in ENVS if any(env in by_variant[v] for v in variant_order)
    ]
    if not envs_present:
        raise RuntimeError("no trajectory data loaded")
    if n_skip:
        print(f"Skipped {n_skip} empty seed run(s)")

    lims = env_axis_limits(by_variant, variant_order, envs_present)
    plotted = clip_trajectories(by_variant, args.tmax) if args.tmax is not None else by_variant
    t_clim = float(args.tmax) if args.tmax is not None else None
    print_coverage(plotted, variant_order, envs_present, env_lims=lims)
    plot_trajectories(
        plotted,
        variant_order,
        envs_present,
        args.output,
        font_size=args.font_size,
        serif=args.serif,
        dpi=args.dpi,
        env_lims=lims,
        t_clim=t_clim,
    )
    for tmax in args.also_tmax or []:
        cropped = clip_trajectories(by_variant, tmax)
        extra = args.output.with_name(f"{args.output.stem}_t{tmax}{args.output.suffix}")
        print(f"\nFirst {tmax + 1} steps (step <= {tmax}):")
        print_coverage(cropped, variant_order, envs_present, env_lims=lims)
        plot_trajectories(
            cropped,
            variant_order,
            envs_present,
            extra,
            font_size=args.font_size,
            serif=args.serif,
            dpi=args.dpi,
            env_lims=lims,
            t_clim=float(tmax),
        )


if __name__ == "__main__":
    main()
