#!/bin/bash
#SBATCH --job-name=fingerprint_train
#SBATCH --partition=compute
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# Usage:
#   sbatch run.sh <exp-name> [--config-dir <dir>] [--set KEY=VALUE ...]
# Examples:
#   sbatch run.sh baseline
#   sbatch run.sh sweep_lr --config-dir config_sweep --set LR=5e-3
EXP_NAME="${1:-default}"

echo "Starting job on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Experiment:   $EXP_NAME"

# -----------------------------------
# Paths
# -----------------------------------
CONTAINER="/data/dayneguy/fingerprint/env/thinkmatch"
WORKDIR="/data/dayneguy/fingerprint"

mkdir -p "$WORKDIR/logs"

# -----------------------------------
# Run training inside container
# -----------------------------------
apptainer exec \
    --nv \
    --bind /general:/general \
    --bind /data:/data \
    --pwd $WORKDIR \
    $CONTAINER \
    python train.py --exp-name "$EXP_NAME" "${@:2}"
