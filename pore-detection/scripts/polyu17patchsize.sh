#!/bin/bash
#SBATCH --job-name=opengait
#SBATCH --output=ekyt.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.ibragimov@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=15gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=3

#module load conda
source ~/.bashrc
conda activate  /home/azim.usf/miniconda3/envs/pores
echo hello

chmod +x scripts/experimentSetUp.sh 
./scripts/experimentSetUp.sh experiments/17PatchSize 


python3 train.py --patchSize 17 --poreRadius 3 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda  --softLabels    >> experiments/17PatchSize/PolyUexp33.log ; wait;
python3 train.py --patchSize 17 --poreRadius 4 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda  --softLabels    >> experiments/17PatchSize/PolyUexp34.log; wait;
python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda  --softLabels    >> experiments/17PatchSize/PolyUexp35.log; wait;
python3 train.py --patchSize 17 --poreRadius 6 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp36.log; wait;

python3 train.py --patchSize 17 --poreRadius 3 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp37.log; wait;
python3 train.py --patchSize 17 --poreRadius 4 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp38.log; wait;
python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp39.log; wait;
python3 train.py --patchSize 17 --poreRadius 6 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp40.log; wait;

python3 train.py --patchSize 17 --poreRadius 3 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp41.log; wait;
python3 train.py --patchSize 17 --poreRadius 4 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp42.log; wait;
python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp43.log; wait;
python3 train.py --patchSize 17 --poreRadius 6 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp44.log; wait;

python3 train.py --patchSize 17 --poreRadius 3 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp45.log; wait;
python3 train.py --patchSize 17 --poreRadius 4 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp46.log; wait;
python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp47.log; wait;
python3 train.py --patchSize 17 --poreRadius 6 --groundTruth dataset --trainingRange 21-110 --validationRange 1-5 --testingRange 6-20 --secondtestingRange 20-20 --maxPooling True --experimentPath experiments/17PatchSize --testStartProbability 0.1 --testStartNMSUnion 0.1 --residual  --boundingBoxSize 17 --device cuda --softLabels    >> experiments/17PatchSize/PolyUexp48.log; wait;
