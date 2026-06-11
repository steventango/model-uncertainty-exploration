#!/bin/bash
# Submit the classic-control grid on Vulcan.
#
# Grid: 10 seeds x 5 envs x {(a=1,b=0),(a=0,b=1)} x {mean,sample} = 200 array tasks.
# Each task requests 1 full L40S GPU (--gres=gpu:l40s:1) for 3 hours.
# Skips completed tasks and any array tasks already queued/running (squeue -n mue-classic).
set -euo pipefail

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ACCOUNT="${SLURM_ACCOUNT:-aip-amw8}"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
    echo "No .venv found. Run: $SCRIPT_DIR/setup.sh" >&2
    exit 1
fi

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
TASKS_TO_SUBMIT="$("$VENV_PYTHON" "$SCRIPT_DIR/grid_tasks.py" "$REPO_ROOT")"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Would run: sbatch --account=$ACCOUNT --export=ALL,REPO_ROOT=$REPO_ROOT --array=<pending> $SCRIPT_DIR/classic_control.sbatch"
  echo ""
  "$VENV_PYTHON" "$SCRIPT_DIR/grid_tasks.py" "$REPO_ROOT" --format summary
  exit 0
fi

if [[ -z "$TASKS_TO_SUBMIT" ]]; then
    echo "Nothing to submit (all tasks complete or already queued/running)."
    exit 0
fi

mkdir -p "/scratch/${USER}/logs"

NUM_TASKS="$(tr ',' '\n' <<< "$TASKS_TO_SUBMIT" | wc -l)"
SBATCH_ARGS=(
    --account="$ACCOUNT"
    --export=ALL,REPO_ROOT="$REPO_ROOT"
    --array="$TASKS_TO_SUBMIT"
    "$SCRIPT_DIR/classic_control.sbatch"
)

job_id="$(sbatch "${SBATCH_ARGS[@]}" | awk '{print $NF}')"
echo "Submitted array job ${job_id} (${NUM_TASKS} tasks, 1 full L40S x 3h each)"
echo "Monitor: squeue -u \$USER -j ${job_id}"
echo "Logs:    /scratch/\$USER/logs/mue-classic-${job_id}_*.out"
