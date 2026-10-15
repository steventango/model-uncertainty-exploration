#!/usr/bin/env bash
# Run one seed of each environment with --debug, sequentially.
set -euo pipefail

OUTPUT_DIR="runs/debug"

ENVS=(
    "Pendulum-v1"
    # "MountainCar-v0"
    # "MountainCarContinuous-v0"
    # "CartPole-v1"
    # "Acrobot-v1"
)

EXTRA_ARGS="${@}"
# 600 steps = 3 Pendulum episodes (one re-plan each); longer envs get fewer.

for ENV in "${ENVS[@]}"; do
    echo "=== $ENV ==="
    uv run python main.py --env "$ENV" --num-seeds 1 --max-steps 600 --debug $EXTRA_ARGS --log-dir "$OUTPUT_DIR/${ENV}/blr" model:blr
    uv run python main.py --env "$ENV" --num-seeds 1 --max-steps 600 --debug $EXTRA_ARGS --log-dir "$OUTPUT_DIR/${ENV}/enn" model:enn
    echo ""
done
