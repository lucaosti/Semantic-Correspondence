#!/bin/bash
# One array for the whole experiment (Libra H100, one GPU per task). Task i < N runs line i+1
# of runs/train_commands.txt, written by the pipeline itself so the arguments are identical;
# task N runs the pipeline, which skips the finished trainings and does sweeps + evaluation:
#   .venv/bin/python scripts/run_pipeline.py --config configs/experiment.yaml \
#       --write-train-commands runs/train_commands.txt
#   N=$(wc -l < runs/train_commands.txt)
#   A=$(sbatch --parsable --array=0-$N scripts/slurm_train_array.sbatch)
#   scontrol update JobId=${A}_$N Dependency=afterok:$(seq -s: -f "${A}_%g" 0 $((N-1)))
# Every line ends with `--resume auto`: resubmitting a failed index continues that run.
#SBATCH --job-name=semcorr-train
#SBATCH --partition=ccstar,machinelearning
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${PWD}"  # scripts/*.py import the repo packages
# torch.compile/Triton builds a C helper with the `module load gcc` (spack 14.2) compiler, whose
# linker cannot find crtbeginS.o/libgcc on the compute nodes: point it at gcc's own lib dir.
if [ -n "${GCC_ROOT:-}" ]; then
  export LIBRARY_PATH="$(dirname "$(find "${GCC_ROOT}/lib/gcc" -name crtbeginS.o | head -1)"):${LIBRARY_PATH:-}"
fi
echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
n=$(wc -l < runs/train_commands.txt)
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${n}" ]; then
  exec .venv/bin/python scripts/run_pipeline.py --config configs/experiment.yaml
fi
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" runs/train_commands.txt)
[ -n "${line}" ] || { echo "no command for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
echo "train.py ${line}"
eval "exec .venv/bin/python scripts/train.py ${line}"
