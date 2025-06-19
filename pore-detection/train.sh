#!/bin/bash
#SBATCH --job-name=AZIM-TRAIN
#SBATCH --output=train.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.ibragimov@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=21gb
#SBATCH --time=72:00:00

echo DPF
python3 dpf.py  --groundTruth dataset --testingRange 1006-1020  --experimentPath experiments/dpf



#echo Dahia
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-90 --validationRange 1001-1005 --testingRange 1006-1020  --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.1 --testEndProbability 0.9 --testStepProbability 0.1 --testStartNMSUnion 0.0 --testStepNMSUnion 0.1  --testEndNMSUnion 0.9  --experimentPath experiments/dahia  --gabriel --device cuda --numberWorkers 8 
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-90 --validationRange 1021-1025 --testingRange 1026-1040  --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.1 --testEndProbability 0.9 --testStepProbability 0.1 --testStartNMSUnion 0.0 --testStepNMSUnion 0.1  --testEndNMSUnion 0.9  --experimentPath experiments/dahia  --gabriel --device cuda --numberWorkers 8 

#echo SU
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-90 --validationRange 1001-1005 --testingRange 1006-1020  --criteriation BCELOSS --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.0 --testEndProbability 1.0  --testStartNMSUnion 0.0   --experimentPath experiments/su  --su --device cuda --numberWorkers 4
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-90 --validationRange 1021-1025 --testingRange 1026-1040  --criteriation BCELOSS --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.0 --testEndProbability 1.0  --testStartNMSUnion 0.0   --experimentPath experiments/su  --su --device cuda --numberWorkers 4
#
#echo Baseline
#python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 1-90 --validationRange 1001-1005 --testingRange 1006-1020  --experimentPath experiments/baseline --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels --numberWorkers 2 --numberFeatures 40 
#python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 1-90 --validationRange 1021-1025 --testingRange 1026-1040  --experimentPath experiments/baseline --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels --numberWorkers 2 --numberFeatures 40
#
#
#echo Dahia
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-700 --validationRange 1001-1005 --testingRange 1006-1020  --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.1 --testEndProbability 0.9 --testStepProbability 0.1 --testStartNMSUnion 0.0 --testStepNMSUnion 0.1  --testEndNMSUnion 0.9  --experimentPath experiments/dahia  --gabriel --device cuda --numberWorkers 8 
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-700 --validationRange 1021-1025 --testingRange 1026-1040  --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.1 --testEndProbability 0.9 --testStepProbability 0.1 --testStartNMSUnion 0.0 --testStepNMSUnion 0.1  --testEndNMSUnion 0.9  --experimentPath experiments/dahia  --gabriel --device cuda --numberWorkers 8 

#echo SU
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-700 --validationRange 1001-1005 --testingRange 1006-1020  --criteriation BCELOSS --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.0 --testEndProbability 1.0  --testStartNMSUnion 0.0   --experimentPath experiments/su  --su --device cuda --numberWorkers 4
#python3 train.py --patchSize 17 --boundingBoxSize 17 --poreRadius 3 --optimizer SGD --learningRate 0.1 --batchSize 256 --groundTruth dataset --trainingRange 1-700 --validationRange 1021-1025 --testingRange 1026-1040  --criteriation BCELOSS --defaultNMS 0 --defaultProb 0.6 --testStartProbability 0.0 --testEndProbability 1.0  --testStartNMSUnion 0.0   --experimentPath experiments/su  --su --device cuda --numberWorkers 4
#
#echo Baseline
#python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 1-700 --validationRange 1001-1005 --testingRange 1006-1020  --experimentPath experiments/baseline --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels --numberWorkers 2 --numberFeatures 40 
#python3 train.py --patchSize 17 --poreRadius 5 --groundTruth dataset --trainingRange 1-700 --validationRange 1021-1025 --testingRange 1026-1040  --experimentPath experiments/baseline --testStartProbability 0.1 --testStartNMSUnion 0.1 --boundingBoxSize 17 --device cuda --softLabels --numberWorkers 2 --numberFeatures 40
