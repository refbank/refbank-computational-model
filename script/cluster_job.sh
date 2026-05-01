#!/bin/bash
#SBATCH --job-name=refbank-fit
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/refbank_%j.out
#SBATCH --error=logs/refbank_%j.err

# Run from the project root: sbatch script/cluster_job.sh
# Adjust --partition, --account, and --time for your cluster.
#
# Expected wall time (lax.scan + GPU):
#   listener fit (5000 steps):        ~30s
#   convention fit × 2 (8000 steps):  ~2min
#   total:                            ~3-4 min
#
# Outputs are written to results/run_<SLURM_JOB_ID>/.
# Load them locally: load_listener_fit / load_convention_fit from
# code/models/literal_listener.py and code/models/conventional_speaker.py.

set -euo pipefail

# ---- cluster-specific: load CUDA and activate venv ----
# Uncomment and adjust for your cluster's module system:
# module load cuda/12.4
# module load python/3.11

source .venv/bin/activate

# ---- create output dir ----
OUTPUT_DIR="results/run_${SLURM_JOB_ID:-local}"
mkdir -p "$OUTPUT_DIR" logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Output dir: $OUTPUT_DIR"
echo "Start: $(date)"

# ---- run pipeline ----
python script/run_pipeline.py \
    --config full_gpu \
    --output-dir "$OUTPUT_DIR"

echo "Done: $(date)"
echo "Results in: $OUTPUT_DIR"
