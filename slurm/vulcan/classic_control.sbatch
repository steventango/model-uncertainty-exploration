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

ENVS=(
    Pendulum-v1
    MountainCar-v0
    MountainCarContinuous-v0
    CartPole-v1
    Acrobot-v1
)
BASE_SEED=0
NUM_SEEDS=30
# Reward weights: exploit-only, explore-only, both.
ALPHAS=(1.0 0.0 1.0)
BETAS=(0.0 1.0 1.0)
PREDICT_MODES=(mean sample)

NUM_PREDICT=${#PREDICT_MODES[@]}
NUM_REWARD=${#ALPHAS[@]}
COMBO_PER_ENV=$((NUM_REWARD * NUM_PREDICT))   # 3 reward weights x 2 predict

# tasks 0-29: 5 envs x 3 reward weights x 2 predict
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

env_idx=$((task_id / COMBO_PER_ENV))
rem=$((task_id % COMBO_PER_ENV))
reward_idx=$((rem / NUM_PREDICT))
predict_idx=$((rem % NUM_PREDICT))

ENV="${ENVS[$env_idx]}"
ALPHA="${ALPHAS[$reward_idx]}"
BETA="${BETAS[$reward_idx]}"
MODE="${PREDICT_MODES[$predict_idx]}"

alpha_tag="${ALPHA/./p}"
beta_tag="${BETA/./p}"
LOG_DIR="runs/classic_grid/${ENV}/a${alpha_tag}_b${beta_tag}/${MODE}"

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
