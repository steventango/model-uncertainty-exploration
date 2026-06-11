#!/bin/bash
# Submit the classic-control grid on Vulcan.
#
# Grid: 5 seeds x 5 envs x {(a=1,b=0),(a=0,b=1)} x {mean,sample} = 100 array tasks.
# Each task requests 1 full L40S GPU (--gres=gpu:l40s:1) for 3 hours.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ACCOUNT="${SLURM_ACCOUNT:-aip-amw8}"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "/scratch/${USER}/logs"
    if [[ ! -d .venv ]]; then
        echo "No .venv found. Run: $SCRIPT_DIR/setup.sh" >&2
        exit 1
    fi
fi

SBATCH_ARGS=(
    --account="$ACCOUNT"
    --export=ALL,REPO_ROOT="$REPO_ROOT"
    "$SCRIPT_DIR/classic_control.sbatch"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Would run: sbatch ${SBATCH_ARGS[*]}"
    echo ""
    echo "Task mapping (first 4 / last 1):"
    python3 - <<'PY'
combos = []
envs = ["Pendulum-v1", "MountainCar-v0", "MountainCarContinuous-v0", "CartPole-v1", "Acrobot-v1"]
seeds = list(range(5))
explore = [(1.0, 0.0), (0.0, 1.0)]
modes = ["mean", "sample"]
for seed in seeds:
    for env in envs:
        for alpha, beta in explore:
            for mode in modes:
                combos.append((len(combos), seed, env, alpha, beta, mode))
for row in combos[:4] + combos[-1:]:
    print(f"  [{row[0]:3d}] seed={row[1]} env={row[2]:28s} a={row[3]} b={row[4]} mode={row[5]}")
print(f"  ... {len(combos)} tasks total")
PY
    exit 0
fi

job_id="$(sbatch "${SBATCH_ARGS[@]}" | awk '{print $NF}')"
echo "Submitted array job ${job_id} (100 tasks, 1 full L40S x 3h each)"
echo "Monitor: squeue -u \$USER -j ${job_id}"
echo "Logs:    /scratch/\$USER/logs/mue-classic-${job_id}_*.out"
