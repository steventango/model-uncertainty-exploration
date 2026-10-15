#!/bin/bash
set -euo pipefail

# Focused comparison PDFs for classic_plan_every_cheap_model_long_variants,
# sized/fonted for direct LaTeX embedding: --scale shrinks the panel canvas so
# --font-size (points, absolute) renders ~WYSIWYG instead of getting shrunk
# further by \includegraphics scale-to-fit. --serif picks a Times New
# Roman-compatible font (falls back to Nimbus Roman / Liberation Serif if the
# real Times New Roman isn't installed).

ROOT=runs/classic_plan_every_cheap_model_long_variants
OUT=runs/compare_plan_every
FONT_SIZE=24
SCALE=0.55
mkdir -p "$OUT"

# G1. MUE (explore_sample_eig: Thompson-sampling rollouts + EIG bonus) vs
# external/true reward (exploit_sample_eig: same TS rollouts, no EIG bonus).
uv run python scripts/plot_learning_curves.py \
  --root "$ROOT" \
  --variants explore_sample_eig exploit_sample_eig \
  --rename-variants \
    explore_sample_eig="EIG Intrinsic Reward" \
    exploit_sample_eig="Extrinsic Reward" \
  --smooth ema \
  --smooth-alpha 0.1 \
  --smooth-adjust \
  --font-size "$FONT_SIZE" \
  --scale "$SCALE" \
  --output "$OUT/G1_mue_vs_true_reward.pdf" \
  || echo "plot G1 skipped (no scalar data yet)"

# G1 exploration: real-env state-space trajectories of the data-collecting
# policy, colored by timestep (same variants / runs as G1).
uv run python scripts/plot_exploration_trajectories.py \
  --root "$ROOT" \
  --variants explore_sample_eig exploit_sample_eig \
  --rename-variants \
    explore_sample_eig="EIG Intrinsic Reward" \
    exploit_sample_eig="Extrinsic Reward" \
  --font-size 20 \
  --also-tmax 100 \
  --output "$OUT/G1_exploration_trajectories.pdf" \
  || echo "plot G1 exploration skipped (no dataset/obs data yet)"
uv run python scripts/plot_exploration_trajectories.py \
  --root "$ROOT" \
  --variants explore_sample_eig exploit_sample_eig \
  --rename-variants \
    explore_sample_eig="EIG Intrinsic Reward" \
    exploit_sample_eig="Extrinsic Reward" \
  --font-size 10 \
  --max-seeds 1 \
  --also-tmax 100 \
  --output "$OUT/G1_exploration_trajectories_1seed.pdf" \
  || echo "plot G1 exploration 1-seed skipped (no dataset/obs data yet)"

# G1b. Ablation ladder: mean-rollout PPO on the model -> + Thompson-sampling
# rollouts -> + EIG exploration bonus (= MUE / explore_sample_eig).
uv run python scripts/plot_learning_curves.py \
  --root "$ROOT" \
  --variants exploit_mean_eig exploit_sample_eig explore_sample_eig \
  --rename-variants \
    exploit_mean_eig=M-PPO \
    exploit_sample_eig=M-PPO+TS \
    explore_sample_eig=M-PPO+TS+EIG \
  --smooth ema \
  --font-size "$FONT_SIZE" \
  --serif \
  --scale "$SCALE" \
  --output "$OUT/G1_ablation_ladder.pdf" \
  || echo "plot G1b skipped (no scalar data yet)"

# G2. MUE (explore_sample_eig, Thompson-sampling rollouts) vs mean rollouts
# under the same explore setting (explore_mean_eig) — isolates rollout mode
# with exploit/explore held fixed.
uv run python scripts/plot_learning_curves.py \
  --root "$ROOT" \
  --variants explore_sample_eig explore_mean_eig \
  --rename-variants \
    explore_sample_eig=MUE \
    explore_mean_eig="MUE, mean rollouts" \
  --smooth ema \
  --font-size "$FONT_SIZE" \
  --serif \
  --scale "$SCALE" \
  --output "$OUT/G2_mue_vs_mean_rollout.pdf" \
  || echo "plot G2 skipped (no scalar data yet)"

# G3. Within MUE (explore_sample_eig) only: the two PPO configs co-trained
# every outer iteration on the same model — "explore" (the variant's own
# alpha/beta+bonus, drives real data collection) vs "eval" (oracle reward,
# no bonus; this is the policy deployed to the real env for eval/mean_return).
# Both are in-model (simulated) training returns, not real-env return.
uv run python scripts/plot_ppo_rollout_returns.py \
  --root "$ROOT" \
  --variant explore_sample_eig \
  --font-size "$FONT_SIZE" \
  --serif \
  --scale "$SCALE" \
  --output "$OUT/G3_mue_ppo_explore_vs_eval.pdf" \
  || echo "plot G3 skipped (no scalar data yet)"
