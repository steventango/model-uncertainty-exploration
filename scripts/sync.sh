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

# A. Does a cheaper update budget recover plan-every quality?
uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every/enn \
    classic_plan_every_budget/full_model_cheap_ppo \
    classic_plan_every_budget/cheap_model_full_ppo \
    classic_plan_every_fast/enn \
  --smooth ema \
  --output runs/compare_plan_every/A_cheaper_budget.png \
  || echo "plot A skipped (no scalar data yet)"

# C. Cheap PPO: does PPO LN and/or higher LR recover full plan-every?
uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every/enn \
    classic_plan_every_cheap_ppo_lr/ln_off_lr_3e-4 \
    classic_plan_every_cheap_ppo_lr/ln_off_lr_1e-3 \
    classic_plan_every_cheap_ppo_lr/ln_off_lr_3e-3 \
    classic_plan_every_cheap_ppo_lr/ln_on_lr_3e-4 \
    classic_plan_every_cheap_ppo_lr/ln_on_lr_1e-3 \
    classic_plan_every_cheap_ppo_lr/ln_on_lr_3e-3 \
  --smooth ema \
  --output runs/compare_plan_every/C_cheap_ppo_ln_lr.png \
  || echo "plot C skipped (no scalar data yet)"

# D. Cheap PPO: does entropy regularization recover full plan-every?
uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every/enn \
    classic_plan_every_cheap_ppo_ent/ent_0 \
    classic_plan_every_cheap_ppo_ent/ent_0p01 \
    classic_plan_every_cheap_ppo_ent/ent_0p1 \
  --smooth ema \
  --output runs/compare_plan_every/D_cheap_ppo_ent.png \
  || echo "plot D skipped (no scalar data yet)"


# B. Does ENN model LayerNorm help? One plot per (model, PPO) budget pair.
uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every/enn \
    classic_plan_every_enn_ln/full_model_ln_full_ppo \
  --output runs/compare_plan_every/B_full_full.png \
  || echo "plot B_full_full skipped (no scalar data yet)"

uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every_budget/cheap_model_full_ppo \
    classic_plan_every_enn_ln/cheap_model_ln_full_ppo \
  --output runs/compare_plan_every/B_cheap_full.png \
  || echo "plot B_cheap_full skipped (no scalar data yet)"

uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every_budget/full_model_cheap_ppo \
    classic_plan_every_enn_ln/full_model_ln_cheap_ppo \
  --output runs/compare_plan_every/B_full_cheap.png \
  || echo "plot B_full_cheap skipped (no scalar data yet)"

uv run python scripts/plot_learning_curves.py \
  --root "${ROOTS[@]}" \
  --variants \
    classic_plan_every_fast/enn \
    classic_plan_every_enn_ln/cheap_model_ln_cheap_ppo \
  --output runs/compare_plan_every/B_cheap_cheap.png \
  || echo "plot B_cheap_cheap skipped (no scalar data yet)"
