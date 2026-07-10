#!/bin/bash
set -euo pipefail

ROOT="vulcan:/home/stang5/scratch/model-uncertainty-exploration/runs"

for exp in classic_plan_every classic_plan_every_fast classic_plan_every_fast_ln; do
  rsync -avz --progress "${ROOT}/${exp}" runs/
  uv run python scripts/plot_learning_curves.py --root "runs/${exp}" \
    || echo "plot skipped for ${exp} (no scalar data yet)"
done
