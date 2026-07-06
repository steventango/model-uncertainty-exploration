#!/usr/bin/env bash
set -euo pipefail

# ENVS=(inverted_pendulum inverted_double_pendulum reacher pusher halfcheetah hopper walker2d swimmer ant)
# ENVS=(swimmer halfcheetah hopper walker2d ant inverted_pendulum inverted_double_pendulum reacher pusher)
ENVS=(halfcheetah)
MODEL="${1:-model:enn}"

for env in "${ENVS[@]}"; do
    echo "=== Running $env ==="
    uv run python main.py --env "$env" "$MODEL"
    echo "=== $env DONE ==="
done
