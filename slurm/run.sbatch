#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/scratch/%u/logs/mue/%x/%j_%a.out
#SBATCH --error=/scratch/%u/logs/mue/%x/%j_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
EXPERIMENT="${EXPERIMENT:?EXPERIMENT not set — submit via python -m slurm submit}"
cd "$REPO_ROOT"

mkdir -p "/scratch/${USER}/logs/mue/${SLURM_JOB_NAME}"

module load python/3.13 cuda/12

if [[ ! -d .venv ]]; then
    echo "Missing .venv in $REPO_ROOT — run slurm/setup.sh first."
    exit 1
fi

VENV="${REPO_ROOT}/.venv/bin/python"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
eval "$("$VENV" -m slurm.prepare_task "$EXPERIMENT" "$task_id")"
echo "experiment=${EXPERIMENT} task=${task_id}: ${MAIN_ARGS}"
exec "$VENV" ${MAIN_ARGS}
