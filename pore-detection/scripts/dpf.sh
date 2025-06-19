#!/bin/bash
#SBATCH --job-name=opengait
#SBATCH --output=ekyt.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.ibragimov@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=15gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=3

#module load conda
source ~/.bashrc
conda activate  /home/azim.usf/miniconda3/envs/pore
echo hello


chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/dpf 

python3 dpf.py  --groundTruth dataset --testingRange 5-20  --experimentPath experiments/dpf
