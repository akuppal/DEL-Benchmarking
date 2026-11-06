#!/bin/bash
# Submit 5-fold CV + test for Chemprop on one GPU.
#
# Usage (from the repo root):
#   mkdir -p logs
#   sbatch submit_chemprop.sh            # Random split
#   sbatch submit_chemprop.sh Cluster    # Cluster split
#
# Finished folds are skipped on rerun, so if the job times out, resubmit the same command.
#SBATCH --job-name=del-chemprop
#SBATCH --partition=condo
#SBATCH --qos=condo-gpu
#SBATCH --account=csd792
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

SPLIT=${1:-Random}

cd "$SLURM_SUBMIT_DIR"
# Activate your environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate del

echo "Job $SLURM_JOB_ID: chemprop on $SPLIT, host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
start=$(date +%s)
python train_chemprop.py --split "$SPLIT" "${@:2}"
echo "Job wall time: $(( $(date +%s) - start ))s"
