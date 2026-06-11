#!/bin/bash
#SBATCH --job-name=mue-classic
#SBATCH --account=aip-amw8
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-299
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
SEEDS=(0 1 2 3 4 5 6 7 8 9)
LEGACY_ALPHAS=(1.0 0.0)
LEGACY_BETAS=(0.0 1.0)
PREDICT_MODES=(mean sample)

LEGACY_NUM_TASKS=200
COMBO_PER_SEED_LEGACY=20   # 5 envs x 2 reward weights x 2 predict
COMBO_PER_ENV_LEGACY=4     # 2 reward weights x 2 predict
COMBO_PER_SEED_BOTH=10     # 5 envs x 2 predict

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

if (( task_id < LEGACY_NUM_TASKS )); then
    seed=$((task_id / COMBO_PER_SEED_LEGACY))
    rem=$((task_id % COMBO_PER_SEED_LEGACY))
    env_idx=$((rem / COMBO_PER_ENV_LEGACY))
    rem2=$((rem % COMBO_PER_ENV_LEGACY))
    reward_idx=$((rem2 / 2))
    predict_idx=$((rem2 % 2))
    ALPHA="${LEGACY_ALPHAS[$reward_idx]}"
    BETA="${LEGACY_BETAS[$reward_idx]}"
else
    sub=$((task_id - LEGACY_NUM_TASKS))
    seed=$((sub / COMBO_PER_SEED_BOTH))
    rem=$((sub % COMBO_PER_SEED_BOTH))
    env_idx=$((rem / 2))
    predict_idx=$((rem % 2))
    ALPHA=1.0
    BETA=1.0
fi

ENV="${ENVS[$env_idx]}"
SEED="${SEEDS[$seed]}"
MODE="${PREDICT_MODES[$predict_idx]}"

alpha_tag="${ALPHA/./p}"
beta_tag="${BETA/./p}"
LOG_DIR="runs/classic_grid/${ENV}/a${alpha_tag}_b${beta_tag}/${MODE}/seed${SEED}"

echo "task=${task_id} env=${ENV} seed=${SEED} alpha=${ALPHA} beta=${BETA} mode=${MODE}"
echo "log_dir=${LOG_DIR}"

"$VENV" main.py \
    --env "$ENV" \
    --seed "$SEED" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --model_env_mode "$MODE" \
    --log_dir "$LOG_DIR"
