#!/bin/bash
#SBATCH --job-name=refbank-fit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/refbank_%j.out
#SBATCH --error=logs/refbank_%j.err

# Run from the project root: sbatch script/cluster_job.sh
#
# Before submitting, pull the container once (store in $GROUP_HOME to avoid quota):
#   apptainer pull $GROUP_HOME/containers/refbank-gpu.sif docker://<dockerhub-user>/refbank-gpu:latest
#
# Then set CONTAINER below to the path of the .sif file.
#
# Expected wall time (lax.scan + GPU):
#   listener fit (5000 steps):       ~30s
#   convention fit × 2 (8000 steps): ~2min
#   total:                           ~3-4 min
#
# Outputs are written to results/run_<SLURM_JOB_ID>/.

set -euo pipefail

CONTAINER="${CONTAINER:-$GROUP_HOME/containers/refbank-gpu.sif}"

if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: container not found at $CONTAINER"
    echo "Pull it with: apptainer pull \$GROUP_HOME/containers/refbank-gpu.sif docker://vboyce/refbank-gpu:latest"
    exit 1
fi

OUTPUT_DIR="results/run_${SLURM_JOB_ID:-local}"
mkdir -p "$OUTPUT_DIR" logs

echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Container:  $CONTAINER"
echo "Output dir: $OUTPUT_DIR"
echo "Start:      $(date)"

# PYTHONPATH makes the project's code/ importable without installing the package
apptainer exec --nv \
    --bind "$(pwd):/project" \
    "$CONTAINER" \
    bash -c "cd /project && PYTHONPATH=/project python script/run_pipeline.py \
        --config full_gpu \
        --no-fetch \
        --output-dir $OUTPUT_DIR"

echo "Done: $(date)"
echo "Results in: $OUTPUT_DIR"
