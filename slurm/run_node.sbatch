#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

EXPERIMENT="${EXPERIMENT:?EXPERIMENT not set}"
TASKS="${TASKS:?TASKS not set (space-separated task ids, one per GPU)}"
cd "${SLURM_SUBMIT_DIR}"

module load python/3.13 cuda/12.9

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

log_dir="${SCRATCH}/logs/mue/${SLURM_JOB_NAME}"
mkdir -p "$log_dir"

gpu=0
pids=()
for task_id in $TASKS; do
    eval "$(.venv/bin/python -m slurm.prepare_task "$EXPERIMENT" "$task_id")"
    echo "experiment=${EXPERIMENT} task=${task_id} gpu=${gpu}: ${MAIN_ARGS}"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python ${MAIN_ARGS} \
        >"${log_dir}/${SLURM_JOB_ID}_${task_id}.out" 2>&1 &
    pids+=($!)
    gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit $status
