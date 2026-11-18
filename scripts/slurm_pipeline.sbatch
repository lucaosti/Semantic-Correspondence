#!/bin/bash
# Run the full pipeline (or any repo command) on one H100 of the Libra `ccstar` partition.
# With the training job array (slurm_train_array.sbatch) it is the final step: trainings are
# found complete and skipped, then the val sweeps and all evaluations run.
# Submit from the repository root, after the one-time setup on the login node
# (.venv, data/, checkpoints/: compute nodes have no internet):
#   mkdir -p runs/slurm && sbatch scripts/slurm_pipeline.sbatch
#   sbatch scripts/slurm_pipeline.sbatch scripts/some_other_script.py --flag
# The pipeline resumes from runs/pipeline_state.json, so a resubmission continues the run.
#SBATCH --job-name=semcorr
#SBATCH --partition=ccstar,machinelearning
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=runs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${PWD}"  # scripts/*.py import the repo packages
# torch.compile/Triton builds a C helper with the `module load gcc` (spack 14.2) compiler, whose
# linker cannot find crtbeginS.o/libgcc on the compute nodes: point it at gcc's own lib dir.
if [ -n "${GCC_ROOT:-}" ]; then
  export LIBRARY_PATH="$(dirname "$(find "${GCC_ROOT}/lib/gcc" -name crtbeginS.o | head -1)"):${LIBRARY_PATH:-}"
fi
echo "job=${SLURM_JOB_ID} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
if [ "$#" -eq 0 ]; then
  set -- scripts/run_pipeline.py --config configs/experiment.yaml
fi
exec .venv/bin/python "$@"
