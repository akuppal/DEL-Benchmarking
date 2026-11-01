#!/bin/bash
# Submit 5-fold CV + test runs for all models as a SLURM array (one task per model).
#
# Usage (from the repo root):
#   mkdir -p logs
#   sbatch submit_cv.sh                  # all models, Random split
#   sbatch submit_cv.sh Cluster          # all models, Cluster split
#   sbatch --array=1,3 submit_cv.sh      # only rf and xgb (indices into MODELS below)
#
# Add --partition/--account (and --gres=gpu:1 for the MLP) as your cluster requires.
#SBATCH --job-name=del-cv
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

MODELS=(mlp rf svm xgb)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SPLIT=${1:-Random}

cd "$SLURM_SUBMIT_DIR"
# Activate your environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate del

echo "Job $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID): $MODEL on $SPLIT, host $(hostname)"
start=$(date +%s)
python train_cv.py --model "$MODEL" --split "$SPLIT"
echo "Job wall time: $(( $(date +%s) - start ))s"
