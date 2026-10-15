#!/usr/bin/env python3
"""Render slides A and B from the rollouts produced by make_algorithm_slides.py.

Slide A: one posterior sample = one imagined plant.
Slide B: 2048 of them, one shared policy acting in all.

y is plant area in cm^2 on a log axis: real units for the audience, while the
model's own log-area spread stays visually symmetric.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_POST = "#1f7a8c"   # posterior / ensemble
C_ACCENT = "#e07a2f" # policy / action
C_GUIDE = "#9aa5ab"

# Slides A and B share axes so they cross-fade cleanly as a slide build.
YLIM = (0.12, 3.0)
FIGSIZE = (8.0, 4.8)


def set_theme(font_size=15):
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": font_size,
        "axes.titlesize": font_size + 3,
        "axes.labelsize": font_size + 1,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_axes(ax, days):
    ax.set_yscale("log")
    ax.set_xlabel("Day")
    ax.set_ylabel("Plant area (cm$^2$)")
    ax.set_xlim(days[0], days[-1])
    ax.set_xticks([0, 4, 8, 12])
    ax.set_ylim(*YLIM)
    ax.set_yticks([0.25, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.25", "0.5", "1.0", "2.0"])
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=220, bbox_inches="tight",
                    facecolor="white" if ext == "png" else "none")
        print("saved", p)
    plt.close(fig)


def slide_a(area, days, out_dir):
    """One sample -> one self-consistent imagined plant."""
    median = np.median(area, axis=0)
    pick = int(np.argmin(np.sum((np.log(area) - np.log(median)) ** 2, axis=1)))
    y = area[pick]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(days, y, color=C_POST, lw=3.0, marker="o", ms=7,
            mfc="white", mec=C_POST, mew=2.0, zorder=3)
    style_axes(ax, days)
    ax.set_title("One sample from the model posterior\n= one imagined plant",
                 loc="left", pad=14)
    ax.annotate("one draw from the posterior:\nthe same imagined plant\nfor all 14 days",
                xy=(days[8], y[8]), xytext=(0.06, 0.80), textcoords="axes fraction",
                color=C_POST, fontsize=13.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_POST, lw=1.6,
                                connectionstyle="arc3,rad=-0.3"))
    save(fig, out_dir, "slideA_one_sample")


def slide_b(area, days, out_dir, n_draw=90, seed=0):
    """2048 imagined plants, one shared policy acting in all of them."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(area.shape[0], size=min(n_draw, area.shape[0]), replace=False)
    median = np.median(area, axis=0)
    lo, hi = np.percentile(area, [5, 95], axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(days, lo, hi, color=C_POST, alpha=0.16, lw=0, zorder=1)
    for i in idx:
        ax.plot(days, area[i], color=C_POST, alpha=0.16, lw=0.9, zorder=2)
    ax.plot(days, median, color=C_POST, lw=3.0, zorder=4)

    style_axes(ax, days)
    ax.set_title("2048 imagined plants\n— one policy trained in all of them",
                 loc="left", pad=14)

    # Policy motif: one pi box feeding actions into every imagined world.
    box = FancyBboxPatch((0.035, 0.845), 0.105, 0.115, transform=ax.transAxes,
                         boxstyle="round,pad=0.012", fc="white",
                         ec=C_ACCENT, lw=2.2, zorder=6)
    ax.add_patch(box)
    ax.text(0.0875, 0.9025, r"$\pi$", transform=ax.transAxes, ha="center",
            va="center", fontsize=19, color=C_ACCENT, zorder=7)
    ax.text(0.175, 0.9025, "one policy, acting in every imagined plant",
            transform=ax.transAxes, ha="left", va="center", fontsize=13,
            color=C_ACCENT, zorder=7)
    for dx, dy in ((0.055, 0.70), (0.135, 0.665), (0.215, 0.625)):
        ax.add_patch(FancyArrowPatch(
            (0.0875, 0.835), (0.0875 + dx, dy), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=11, color=C_ACCENT, lw=1.4,
            alpha=0.75, connectionstyle="arc3,rad=-0.25", zorder=6))
    ax.text(0.985, 0.055, "each curve = one sampled model     shaded = 5–95%",
            transform=ax.transAxes, fontsize=12.5, color=C_POST,
            va="bottom", ha="right")
    save(fig, out_dir, "slideB_many_samples")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="figures/slides")
    ap.add_argument("--font-size", type=int, default=15)
    args = ap.parse_args()

    d = np.load(os.path.join(args.out_dir, "rollouts.npz"))
    area = np.exp(np.asarray(d["areas"]))  # log-area -> cm^2
    days = np.arange(area.shape[1])

    set_theme(args.font_size)
    slide_a(area, days, args.out_dir)
    slide_b(area, days, args.out_dir)


if __name__ == "__main__":
    main()
