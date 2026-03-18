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

echo "Starting job on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# -----------------------------------
# Paths
# -----------------------------------
CONTAINER="/data/hot/dayneguy/fingerprint/env/thinkmatch"
WORKDIR="/data/hot/dayneguy/fingerprint"

# Ensure we run from project directory
cd $WORKDIR

# -----------------------------------
# Run training inside container
# -----------------------------------
apptainer shell     --nv      --bind /data:/data      --pwd $WORKDIR $CONTAINER 
# python train.py
