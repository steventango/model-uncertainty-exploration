"""Thesis figures for the gated ("reduce energy if growth is sufficient") case study.

Figure 1: the learned policy - commanded intensity over (day, log area), with the
          0.9x area gate the reward is defined against and the trajectory the
          deployed chamber actually followed.
Figure 2: the deployment result - area against the gate, commanded intensity,
          cumulative energy, and the growth/energy summary against two baselines.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.append("/home/steven/Github/model-uncertainty-exploration")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from networks import ActorCritic
from normalization import NormalizeVecObs
from plant_env import _BASELINE_AREA, _MASK_F


# plant-data helpers, inlined: importing them would shadow this repo's own config module.
def drop_bad_initial_area(per_plant, area_col="a0", max_rel=4.0):
    """Drop plants whose day-0 area is >max_rel x the zone median (segmentation failure)."""
    rel = pl.col(area_col) / pl.col(area_col).median().over(["experiment", "zone"])
    tagged = per_plant.with_columns(rel.alias("_rel"))
    for row in tagged.filter(pl.col("_rel") > max_rel).select(
        "experiment", "zone", "plant_id", area_col, "_rel"
    ).iter_rows():
        print("dropping E%d Z%d plant %s: day-0 area %.2f = %.0fx zone median"
              % (row[0], row[1], row[2], row[3], row[4]))
    return tagged.filter(pl.col("_rel") <= max_rel).drop("_rel")


def set_theme(style="white"):
    import seaborn as sns

    sns.set_theme(style=style)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })


def savefig(fig, path, dpi=200, bbox_inches="tight", **kw):
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kw)
    plt.close(fig)
    print(f"Saved -> {path}")

CKPT = "/home/steven/Github/plant-rl/checkpoints/E21/P1/masked_log/checkpoint"
PARQUET = "/data/plant-rl/offline/v29/all-daily-v29.parquet"
OUT = "/home/steven/thesis_figs"

TARGET = (21, 2)          # gated policy, deterministic action
SAME_BATCH = (21, 11)     # constant baseline, same plant batch
GATE_REF = (18, 11)       # constant baseline the 0.9x gate was calibrated on
END = 14
A_MIN, A_MAX = 0.4, 1.3

COLOR_POLICY = "#eb6834"
COLOR_SAME = "#2a78d6"
COLOR_REF = "#1baf7a"
COLOR_BAR = "#7a5195"
TEXT = "#33322e"

CPU = jax.devices("cpu")[0]


# --------------------------------------------------------------------------- policy
def restore(module, path):
    g, s = nnx.split(module)
    s = ocp.StandardCheckpointer().restore(os.path.abspath(path), target=s)
    return nnx.merge(g, s)


def load_policy():
    with jax.default_device(CPU):
        net = restore(
            ActorCritic(2, 1, 64, activation="tanh", use_layer_norm=False, rngs=nnx.Rngs(0)),
            os.path.join(CKPT, "ppo_eval", "network"),
        )
        obs_norm = restore(NormalizeVecObs(jnp.zeros(2)), os.path.join(CKPT, "ppo_eval", "obs_norm"))
    return net, obs_norm


def policy_actions(net, obs_norm, log_area, days):
    """Commanded intensity on the (day, log-area) grid, as the chamber would apply it."""
    la = np.asarray(log_area, dtype=np.float32)
    out = np.empty((len(days), len(la)), dtype=np.float32)
    with jax.default_device(CPU):
        for i, d in enumerate(days):
            obs = jnp.stack([jnp.asarray(la), jnp.full(la.shape, float(d), dtype=jnp.float32)], -1)
            pi, _ = net(obs_norm.normalize(obs))
            out[i] = np.clip(np.asarray(pi.mean())[:, 0], A_MIN, A_MAX)
    return out


# ----------------------------------------------------------------------------- data
def load_data():
    return pl.read_parquet(
        PARQUET,
        columns=["experiment", "zone", "day", "plant_id", "clean_area", "energy", "intensity"],
    )


# E18 Z11 day 2 carries an all-zero clean_area (imaging gap; the gate table interpolates it).
# Those rows must stay out of every area statistic, but the lights still ran that day, so
# energy and intensity are aggregated over the unscreened frame.
AREA_OK = pl.col("clean_area").is_not_null() & (pl.col("clean_area") > 0)


def daily(df, exp, zone):
    z = df.filter((pl.col("experiment") == exp) & (pl.col("zone") == zone) & (pl.col("day") <= END))
    return (
        z.group_by("day")
        .agg([
            pl.col("clean_area").filter(AREA_OK).log().mean().alias("mean_log_area"),
            pl.col("clean_area").filter(AREA_OK).mean().alias("mean_area"),
            pl.col("energy").mean().alias("energy"),
            pl.col("intensity").drop_nans().drop_nulls().mean().alias("intensity"),
        ])
        .sort("day")
    )


def per_plant(df, exp, zone):
    z = df.filter((pl.col("experiment") == exp) & (pl.col("zone") == zone))
    a0 = z.filter(pl.col("day") == 0).select(
        ["experiment", "zone", "plant_id", pl.col("clean_area").alias("a0")]
    )
    aT = z.filter(pl.col("day") == END).select(["plant_id", pl.col("clean_area").alias("aT")])
    m = a0.join(aT, on="plant_id", how="inner").drop_nulls()
    m = drop_bad_initial_area(m, "a0")
    return m.with_columns((pl.col("aT").log() - pl.col("a0").log()).alias("loggrowth"))


def daily_area_ci(df, exp, zone, n_boot=10000):
    """Per-day BCa interval on the cohort mean area, resampling the plants.

    The gate baseline is a fixed table, not an estimate, so only the compartment's
    own plants are resampled.
    """
    from scipy.stats import bootstrap

    z = df.filter((pl.col("experiment") == exp) & (pl.col("zone") == zone)
                  & (pl.col("day") <= END))
    days, lo, hi = [], [], []
    for d in sorted(z["day"].unique().to_list()):
        a = z.filter((pl.col("day") == d) & AREA_OK)["clean_area"].to_numpy()
        if a.size < 3:
            continue
        r = bootstrap((a,), np.mean, n_resamples=n_boot, method="BCa", random_state=2026)
        days.append(d)
        lo.append(r.confidence_interval.low)
        hi.append(r.confidence_interval.high)
    return np.asarray(days), np.asarray(lo), np.asarray(hi)


def gate_pass_rate(df, exp, zone):
    """Fraction of plant-days at or above the 0.9x area gate, day 1..END."""
    z = df.filter(
        (pl.col("experiment") == exp) & (pl.col("zone") == zone)
        & (pl.col("day") >= 1) & (pl.col("day") <= END)
    ).filter(AREA_OK).select(["day", "plant_id", "clean_area"])
    gate = np.asarray(_MASK_F * _BASELINE_AREA)
    d = z["day"].to_numpy().astype(int)
    return float(np.mean(z["clean_area"].to_numpy() >= gate[np.clip(d, 0, len(gate) - 1)]))


# -------------------------------------------------------------------------- figure 1
def figure_policy(net, obs_norm):
    set_theme()
    plt.rcParams.update({
        "font.size": 18, "axes.labelsize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
    })
    days = np.arange(0, 15)
    la = np.linspace(np.log(0.15), np.log(9.0), 400)
    A = policy_actions(net, obs_norm, la, days)

    gate = np.log(_MASK_F * np.asarray(_BASELINE_AREA))
    gate_next = gate[np.clip(days + 1, 0, len(gate) - 1)]

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.6), layout="constrained")

    # Area on a log axis rather than log-area on a linear one: the same surface,
    # but read directly in the units the plants are measured in.
    area = np.exp(la)
    mesh = ax.pcolormesh(
        days, area, 100 * A.T, cmap="cividis", vmin=100 * A_MIN, vmax=100 * A_MAX,
        shading="nearest", rasterized=True
    )
    ax.plot(days, np.exp(gate_next), color="w", lw=2.6, ls="--", zorder=4)
    # A proper log axis: 1-2-5 majors printed without trailing zeros, decade
    # minors drawn as short ticks so the spacing reads as logarithmic, and the
    # view trimmed to the range the gate and the cohort actually occupy.
    tick = plt.matplotlib.ticker
    ax.set_yscale("log")
    ax.set_ylim(0.2, 6.0)
    ax.yaxis.set_major_locator(tick.FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(tick.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                               numticks=100))
    ax.yaxis.set_minor_formatter(tick.NullFormatter())
    ax.tick_params(axis="y", which="major", length=9, width=1.5)
    ax.tick_params(axis="y", which="minor", length=5, width=1.1)
    ax.set_xlabel("Day")
    ax.set_ylabel(r"Area (cm$^2$)")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Action " + r"($\mu$mol$\,$m$^{-2}\,$s$^{-1}$)", fontsize=18)
    cb.set_ticks([40, 70, 100, 130])
    cb.ax.tick_params(labelsize=17)

    savefig(fig, os.path.join(OUT, "gated_policy.pdf"))


# -------------------------------------------------------------------------- figure 2
def figure_results(df, tgt, same, ref, stats):
    set_theme()
    # This figure is three panels wide and is included at \linewidth, so it is
    # scaled down about twice as hard as the single-panel policy figure. The
    # point sizes are set large to compensate: what matters is pt per inch of
    # figure width, not pt on its own.
    plt.rcParams.update({
        "font.size": 24, "axes.labelsize": 26,
        "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 21,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.4), layout="constrained",
                             width_ratios=[1.0, 1.0, 1.42])
    baseline = np.asarray(_BASELINE_AREA)

    # (a) cohort mean area as a percentage of the gate baseline. The gate-reference zone
    # sits at 100% by construction (the baseline table is its own per-day mean), so only
    # the two policies under comparison are drawn.
    ax = axes[0]
    for frame, key, color, lab in ((same, SAME_BATCH, COLOR_SAME, "Baseline"),
                                   (tgt, TARGET, COLOR_POLICY, "Learned")):
        d = frame["day"].to_numpy().astype(int)
        pct = 100 * frame["mean_area"].to_numpy() / baseline[np.clip(d, 0, len(baseline) - 1)]
        cd, clo, chi = daily_area_ci(df, *key)
        cb_ = baseline[np.clip(cd.astype(int), 0, len(baseline) - 1)]
        ax.fill_between(cd, 100 * clo / cb_, 100 * chi / cb_, color=color, alpha=0.22,
                        lw=0, zorder=2)
        ax.plot(d, pct, color=color, lw=2.8, label=lab, zorder=3)
    # reference levels are drawn but kept out of the legend
    ax.axhline(100, color="0.45", lw=1.1, ls=":")
    ax.axhline(90, color="0.2", lw=1.6, ls="--")
    ax.set_xlim(0, END)
    ax.set_xticks([0, 7, 14])
    ax.set_ylim(80, max(175, 5 + 100 * (same["mean_area"].to_numpy()
                / baseline[np.clip(same["day"].to_numpy().astype(int), 0, len(baseline) - 1)]).max()))
    # 90 is the gate and 100 the reference cohort, so both get a tick
    ax.set_yticks([90, 100, 120, 140, 160])
    ax.set_xlabel("Day")
    ax.set_ylabel("Relative Area (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)

    # (b) the schedule each zone actually ran, in PPFD. No legend: (a) already carries it.
    ax = axes[1]
    ax.step(same["day"], 100 * same["intensity"], where="mid", color=COLOR_SAME, lw=2.6)
    ax.step(tgt["day"], 100 * tgt["intensity"], where="mid", color=COLOR_POLICY, lw=3.0)
    ax.set_xlim(0, END)
    ax.set_xticks([0, 7, 14])
    ax.set_ylim(32, 144)
    ax.set_xlabel("Day")
    ax.set_ylabel("Action\n" + r"($\mu$mol$\,$m$^{-2}\,$s$^{-1}$)", fontsize=22)
    ax.spines[["top", "right"]].set_visible(False)

    # (c) whole-run summary. Energy is one measured per-zone total that lands within half
    # a point of the same figure against either baseline, so it is drawn once and carries
    # no interval. Growth is drawn against both baselines with a 95% BCa bootstrap
    # interval over the plants in the compartment.
    ax = axes[2]
    e_pct = 100 * (0.5 * (stats["e_same"] + stats["e_ref"]) - 1)
    vals = [e_pct, 100 * (stats["g_same"] - 1), 100 * (stats["g_ref"] - 1)]
    labels = ["Energy",
              "Growth\nrelative to\nconcurrent\nbaseline",
              "Growth\nrelative to\nhistorical\nbaseline"]
    x = np.arange(len(vals))
    ax.axhline(0, color="0.4", lw=1.0)
    ax.bar(x, vals, 0.6, color=COLOR_BAR)
    # interval on the growth bars only: energy is a measured total, not an estimate
    ax.errorbar(x[1:], vals[1:], yerr=np.array([
        [100 * (stats["g_same"] - stats["g_same_lo"]), 100 * (stats["g_ref"] - stats["g_ref_lo"])],
        [100 * (stats["g_same_hi"] - stats["g_same"]), 100 * (stats["g_ref_hi"] - stats["g_ref"])],
    ]), fmt="none", capsize=9, lw=3.2, capthick=3.2, ecolor=TEXT)
    ax.set_xticks(x, labels, fontsize=20)
    ax.set_ylim(min(vals) - 12, 8)
    ax.set_ylabel("Relative Change (%)")
    ax.spines[["top", "right"]].set_visible(False)

    print("(c) energy %+.1f%% (same-batch %+.1f%%, gate-ref %+.1f%%)"
          % (e_pct, 100 * (stats["e_same"] - 1), 100 * (stats["e_ref"] - 1)))
    print("(c) growth vs same-experiment %+.1f%% [%+.1f, %+.1f]"
          % (vals[1], 100 * (stats["g_same_lo"] - 1), 100 * (stats["g_same_hi"] - 1)))
    print("(c) growth vs gate-reference  %+.1f%% [%+.1f, %+.1f]"
          % (vals[2], 100 * (stats["g_ref_lo"] - 1), 100 * (stats["g_ref_hi"] - 1)))

    savefig(fig, os.path.join(OUT, "gated_results.pdf"))


def main():
    from scipy.stats import bootstrap

    df = load_data()
    tgt, same, ref = (daily(df, *k) for k in (TARGET, SAME_BATCH, GATE_REF))

    def energy_total(frame):
        return float(frame.filter(pl.col("day") >= 1)["energy"].sum())

    pt, ps, pr = (per_plant(df, *k) for k in (TARGET, SAME_BATCH, GATE_REF))

    def ratio(a, b):
        st = lambda x, y: np.mean(x) / np.mean(y)
        r = bootstrap((a, b), st, n_resamples=10000, method="BCa", random_state=2026, vectorized=False)
        return st(a, b), r.confidence_interval.low, r.confidence_interval.high

    g_same, g_same_lo, g_same_hi = ratio(pt["loggrowth"].to_numpy(), ps["loggrowth"].to_numpy())
    g_ref, g_ref_lo, g_ref_hi = ratio(pt["loggrowth"].to_numpy(), pr["loggrowth"].to_numpy())
    stats = dict(
        g_same=g_same, g_same_lo=g_same_lo, g_same_hi=g_same_hi,
        g_ref=g_ref, g_ref_lo=g_ref_lo, g_ref_hi=g_ref_hi,
        e_same=energy_total(tgt) / energy_total(same),
        e_ref=energy_total(tgt) / energy_total(ref),
    )

    print("--- verification -------------------------------------------------")
    print("E18 Z11 mean area by day (data) :", " ".join("%.3f" % v for v in ref["mean_area"].to_numpy()))
    print("_BASELINE_AREA (policy gate)    :", " ".join("%.3f" % v for v in np.asarray(_BASELINE_AREA)))
    print("gated-policy intensity by day   :", " ".join("%.2f" % v for v in tgt["intensity"].to_numpy()))
    print("same-batch intensity by day     :", " ".join("%.2f" % v for v in same["intensity"].to_numpy()))
    print("energy kWh  gated / same / ref  : %.2f / %.2f / %.2f"
          % (energy_total(tgt) / 1000, energy_total(same) / 1000, energy_total(ref) / 1000))
    print("gate pass rate (plant-days) gated / same / ref : %.3f / %.3f / %.3f"
          % (gate_pass_rate(df, *TARGET), gate_pass_rate(df, *SAME_BATCH), gate_pass_rate(df, *GATE_REF)))
    print("log-growth ratio vs same-batch  : %.3f [%.3f, %.3f]" % (g_same, g_same_lo, g_same_hi))
    print("log-growth ratio vs gate-ref    : %.3f [%.3f, %.3f]" % (g_ref, g_ref_lo, g_ref_hi))
    print("energy ratio vs same / vs ref   : %.3f / %.3f" % (stats["e_same"], stats["e_ref"]))
    print("n plants gated/same/ref         : %d / %d / %d" % (pt.height, ps.height, pr.height))

    net, obs_norm = load_policy()
    figure_policy(net, obs_norm)
    figure_results(df, tgt, same, ref, stats)


if __name__ == "__main__":
    main()
