#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

module load python/3.13 cuda/12

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; install with: pip install --user uv"
    exit 1
fi

uv sync
echo "Done. .venv is ready at $REPO_ROOT/.venv"
