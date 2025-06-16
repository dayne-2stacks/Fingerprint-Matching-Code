#!/usr/bin/env python

from pathlib import Path
import cv2
import numpy as np

import torch

import os

from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import TestDataset, get_dataloader
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda

from utils.models_sl import load_model
from utils.visualize import visualize_match, visualize_stochastic_matrix, to_grayscale_cv2_image
from src.evaluation_metric import matching_accuracy
from utils.matching import build_matches





dataset_len = 640

# File paths
train_root = 'dataset/Synthetic'
OUTPUT_PATH = "result1"
PRETRAINED_PATH = "result1/params/best_model.pt" 


test_bm =  L3SFV2AugmentedBenchmark(
    sets='test',
    obj_resize=(320, 240),
    train_root=train_root,
    filter="inclusion"
)



test_dataset = TestDataset("L3SFV2Augmented", test_bm, dataset_len, True, None, "2GM")
print(f"Test dataset length: {len(test_dataset)}")

test_dataloader = get_dataloader(test_dataset, shuffle=True, fix_seed=False)

# =====================================================
# Model, Loss, and Device Setup
# =====================================================
model = Net(regression=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# =====================================================
# Checkpoint Loading (if start_epoch > 0)
# =====================================================
if len(PRETRAINED_PATH) > 0:
    model_path = PRETRAINED_PATH


if os.path.exists(model_path):
    print(f"Loading model weights from {model_path} before training loop...")
    load_model(model, model_path)
            


# =====================================================
#  Evaluate on a Sample
# =====================================================
single_sample = next(iter(test_dataloader))
single_sample = data_to_cuda(single_sample)
print(single_sample.keys())


model.eval()
with torch.no_grad():
    print("Running model inference on a single sample...")
    outputs = model(single_sample)
    
print("Model outputs keys:", outputs.keys())
acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
if isinstance(acc, torch.Tensor):
    if acc.numel() > 1:  # Check if tensor has multiple elements
        acc = acc.mean().item()  # Take the mean before converting to scalar
    else:
        acc = acc.item()
        
print("Matching accuracy:", acc)

                
    
# Explicitly select the first sample from the batch
if 'Ps' in single_sample:
    kp0 = single_sample['Ps'][0][0].cpu().numpy()
    kp1 = single_sample['Ps'][1][0].cpu().numpy()
    print("Ps in sample")
else:
    kp0 = np.array([[100, 100], [150, 150], [200, 200]])
    kp1 = np.array([[110, 110], [160, 160], [210, 210]])

print("Number of keypoints in image0 (kp0):", len(kp0))
print("Number of keypoints in image1 (from kp1):", kp1.shape[0])

# Ensure keypoints lists for OpenCV are correctly formed:
cv2_kp0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp0]
cv2_kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp1]

ds_mat = outputs["ds_mat"].cpu().numpy()[0]
per_mat = outputs["perm_mat"].cpu().numpy()[0]

matches = build_matches(ds_mat, per_mat)
    
print(len(single_sample["images"]))

if "id_list" in single_sample:
    img0 = single_sample["images"][0][0]
    img1 = single_sample["images"][1][0]
else:
    img0 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg")
    img1 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg")
    print("Using fallback image paths.")




img0= to_grayscale_cv2_image(img0)
img1 = to_grayscale_cv2_image(img1)

visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/")
visualize_stochastic_matrix(ds_mat, "ds_mat")
print("Accuracy: ", acc)

# Add final visualizations to TensorBoard
if len(matches) > 0:
    match_path = f"photos/final_match.jpg"
    visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/", filename="final_match")
    
    if os.path.exists(match_path):
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)


print("Accuracy: ", acc)
