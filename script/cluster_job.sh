#!/bin/bash
#SBATCH --job-name=refbank-fit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --output=logs/refbank_%j.out
#SBATCH --error=logs/refbank_%j.err

# Run from the project root: sbatch script/cluster_job.sh
#
# Before submitting, build the container sandbox once (store in $GROUP_HOME to avoid quota):
#   apptainer build --sandbox $GROUP_HOME/containers/refbank-gpu/ docker://vboyce/refbank-gpu:latest
#
# Expected wall time (lax.scan + GPU):
#   listener fit (5000 steps):         ~30s
#   convention fit × 2 (8000 steps):   ~2min
#   eval: 20 splits × listener fit:    ~10min
#   total:                             ~15 min
#
# Outputs are written to results/run_<SLURM_JOB_ID>/ and results/eval_listener_cv.csv.

set -euo pipefail

CONTAINER="${CONTAINER:-$GROUP_HOME/containers/refbank-gpu}"

if [[ ! -d "$CONTAINER" ]]; then
    echo "ERROR: container sandbox not found at $CONTAINER"
    echo "Build it with: apptainer build --sandbox \$GROUP_HOME/containers/refbank-gpu/ docker://vboyce/refbank-gpu:latest"
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

echo "Pipeline done: $(date)"
echo "Results in: $OUTPUT_DIR"

echo "Running listener cross-validation (20 splits × 5000 steps)..."
apptainer exec --nv \
    --bind "$(pwd):/project" \
    "$CONTAINER" \
    bash -c "cd /project && PYTHONPATH=/project python script/run_eval.py \
        --n-splits 20 \
        --n-steps 5000 \
        --seed 0"

echo "Eval done: $(date)"
echo "CV results in: results/eval_listener_cv.csv"
