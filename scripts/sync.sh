#!/bin/bash
set -euo pipefail

ROOT="vulcan:/home/stang5/scratch/model-uncertainty-exploration/runs"

EXPS=(
  classic_plan_every
  classic_plan_every_fast
  classic_plan_every_fast_ln
  classic_plan_every_budget
  classic_plan_every_enn_ln
  classic_plan_every_cheap_ppo_lr
  classic_plan_every_cheap_ppo_ent
  classic_plan_every_cheap_model_long
  classic_plan_every_cheap_model_long_squash
  classic_plan_every_cheap_model_long_variants
)

for exp in "${EXPS[@]}"; do
  rsync -avz --progress "${ROOT}/${exp}" runs/
  # uv run python scripts/plot_learning_curves.py --root "runs/${exp}" \
  #   || echo "plot skipped for ${exp} (no scalar data yet)"
done

ROOTS=()
for exp in "${EXPS[@]}"; do
  ROOTS+=("runs/${exp}")
done
mkdir -p runs/compare_plan_every


# E. ENN (blr_enn) vs long cheap-model + full-PPO run.
uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" runs/blr_enn \
  --variants \
    blr_enn/enn \
    classic_plan_every_cheap_model_long/cheap_model_full_ppo \
  --smooth ema \
  --output runs/compare_plan_every/E_cheap_model_long.png \
  || echo "plot E skipped (no scalar data yet)"

# G. Exploit/explore x mean/sample (eig bonus), cheap-model + full-PPO budget, 1000 steps.
uv run python scripts/plot_learning_curves.py \
  --root runs/classic_plan_every_cheap_model_long_variants \
  --variants \
    exploit_mean_eig \
    exploit_sample_eig \
    explore_sample_eig \
    explore_mean_eig \
  --smooth ema \
  --font-size 12 \
  --serif \
  --output runs/compare_plan_every/G_cheap_model_long_variants.pdf \
  || echo "plot G skipped (no scalar data yet)"
