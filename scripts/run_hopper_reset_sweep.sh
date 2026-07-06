#!/usr/bin/env bash
set -euo pipefail

# Run hopper across the three model-env reset sources (env / buffer / init) and
# three model-env rollout-length settings (default / 1 / 10), then plot the
# combined learning curves with scripts/plot_learning_curves.py.
#
# Overridable via env vars:
#   ENV=hopper  MODEL=model:enn  SEEDS=3  NUM_ROLLOUTS=20  ROOT=runs/hopper_reset_sweep
#
# Usage:
#   scripts/run_hopper_reset_sweep.sh

ENV="${ENV:-hopper}"
MODEL="${MODEL:-model:enn}"
SEEDS="${SEEDS:-3}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-20}"
ROOT="${ROOT:-runs/hopper_reset_sweep}"

RESET_SOURCES=(buffer init env)
# "default" is a sentinel meaning: do not pass --rollout-length (use env horizon).
ROLLOUT_LENGTHS=(1 10 default)

cd "$(dirname "$0")/.."
mkdir -p "$ROOT"

for src in "${RESET_SOURCES[@]}"; do
    for rl in "${ROLLOUT_LENGTHS[@]}"; do
        tag="${src}_rl-${rl}"
        log_dir="${ROOT}/${tag}"
        label="${src} rl=${rl}"

        rl_arg=()
        if [[ "$rl" != "default" ]]; then
            rl_arg=(--rollout-length "$rl")
        fi

        echo "=== Running ${tag} (env=${ENV}, seeds=${SEEDS}, rollouts=${NUM_ROLLOUTS}) ==="
        uv run python main.py \
            --env "$ENV" \
            --alpha 1.0 \
            --reset-source "$src" \
            "${rl_arg[@]}" \
            --num-seeds "$SEEDS" \
            --num-rollouts "$NUM_ROLLOUTS" \
            --label "$label" \
            --log-dir "$log_dir" \
            "$MODEL"
        echo "=== ${tag} DONE ==="
        echo "=== Plotting learning curves ==="
        uv run python scripts/plot_learning_curves.py \
            --root "$ROOT" \
            --output "${ROOT}/learning_curves.png"
        echo "Plot: ${ROOT}/learning_curves.png ==="
    done
done

echo "=== Sweep complete. ==="
