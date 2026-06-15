#!/bin/bash
#SBATCH --job-name=mue-classic
#SBATCH --account=aip-amw8
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-29
#SBATCH --output=/scratch/%u/logs/mue-classic-%A_%a.out
#SBATCH --error=/scratch/%u/logs/mue-classic-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$REPO_ROOT"

mkdir -p "/scratch/${USER}/logs"

module load python/3.13 cuda/12

if [[ ! -d .venv ]]; then
    echo "Missing .venv in $REPO_ROOT — run slurm/vulcan/setup.sh first."
    exit 1
fi

VENV="${REPO_ROOT}/.venv/bin/python"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Decode this array task via the single source of truth (submit.py).
# Sets ENV, ALPHA, BETA, MODE, BASE_SEED, NUM_SEEDS, LOG_DIR.
# SLURM runs a spooled copy of this script, so resolve submit.py via REPO_ROOT.
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
eval "$("$VENV" "${REPO_ROOT}/slurm/vulcan/submit.py" env "$task_id")"

echo "task=${task_id} env=${ENV} base_seed=${BASE_SEED} num_seeds=${NUM_SEEDS} alpha=${ALPHA} beta=${BETA} mode=${MODE}"
echo "log_dir=${LOG_DIR}"

"$VENV" main.py \
    --env "$ENV" \
    --seed "$BASE_SEED" \
    --num_seeds "$NUM_SEEDS" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --model_env_mode "$MODE" \
    --log_dir "$LOG_DIR"
