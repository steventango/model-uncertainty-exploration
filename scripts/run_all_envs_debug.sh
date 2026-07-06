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

for ENV in "${ENVS[@]}"; do
    echo "=== $ENV ==="
    uv run python main.py --env "$ENV" --num-seeds 1 --num-rollouts 3 --debug $EXTRA_ARGS --log-dir "$OUTPUT_DIR/${ENV}/blr" model:blr
    uv run python main.py --env "$ENV" --num-seeds 1 --num-rollouts 3 --debug $EXTRA_ARGS --log-dir "$OUTPUT_DIR/${ENV}/enn" model:enn
    echo ""
done
