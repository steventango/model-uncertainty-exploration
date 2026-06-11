#!/bin/bash
#SBATCH --job-name=mue-classic
#SBATCH --account=aip-amw8
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-99
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

cp -R .venv "$SLURM_TMPDIR/venv"
VENV="$SLURM_TMPDIR/venv/bin/python"

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
SEEDS=(0 1 2 3 4)
EXPLORE_ALPHAS=(1.0 0.0)
EXPLORE_BETAS=(0.0 1.0)
PREDICT_MODES=(mean sample)

COMBO_PER_SEED=20   # 5 envs x 2 explore x 2 predict
COMBO_PER_ENV=4     # 2 explore x 2 predict

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

seed=$((task_id / COMBO_PER_SEED))
rem=$((task_id % COMBO_PER_SEED))
env_idx=$((rem / COMBO_PER_ENV))
rem2=$((rem % COMBO_PER_ENV))
explore_idx=$((rem2 / 2))
predict_idx=$((rem2 % 2))

ENV="${ENVS[$env_idx]}"
SEED="${SEEDS[$seed]}"
ALPHA="${EXPLORE_ALPHAS[$explore_idx]}"
BETA="${EXPLORE_BETAS[$explore_idx]}"
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
