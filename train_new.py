#!/usr/bin/env python
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.optim as optim
from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from src.loss_func import PermutationLoss
from utils.models_sl import save_model, load_model
from src.parallel import DataParallel
from utils.matching import build_matches

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -------------------------------
# Parameters & Hyperparameters
# -------------------------------
dataset_len = 640
LR = 2.e-5              # Learning rate for joint training (stage 3)
BACKBONE_LR = 2.e-5       # Learning rate for CNN and GM network in stage 1
K_LR = 2.e-5              # Learning rate for AFA module warm-up (stage 2)
# Epochs per stage
stage1_epochs = 4  # Train CNN + GM network (AFA modules frozen)
stage2_epochs = 4   # Train AFA modules (warm-up)
stage3_epochs = 10  # Joint training

# -------------------------------
# Dataset and Dataloader
# -------------------------------
benchmark = L3SFV2AugmentedBenchmark(
    sets='train',
    obj_resize=(320, 240),
    train_root='/green/data/L3SF_V2/L3SF_V2_Augmented'
)
print(benchmark.get_path("R1_47_right_loop_aug_1"))

image_dataset = GMDataset("L3SFV2Augmented", benchmark, dataset_len, True, None, "2GM")
dataloader = get_dataloader(image_dataset, shuffle=True, fix_seed=True)

# -------------------------------
# Model, Loss, and Device Setup
# -------------------------------
model = Net()
criterion = PermutationLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# Uncomment if you have multiple GPUs
# model = DataParallel(model, device_ids=[0,1])

# -------------------------------
# Checkpoint directory
# -------------------------------
checkpoint_path = Path("result") / 'params'
checkpoint_path.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Placeholder sample (for debugging, adjust as needed)
# -------------------------------
single_sample = next(iter(dataloader))
single_sample = data_to_cuda(single_sample)  # move sample to GPU if available
print("ID list:", single_sample.get("id_list", "Not provided"))
print("Keypoint count (ns):", single_sample["ns"])

# -------------------------------
# Training Script with 3 Stages
# -------------------------------

# Stage 1: Train CNN + GM network only.
# Freeze AFA modules (encoder_k, final_row, final_col) to avoid error from untrained AFA.
for param in model.encoder_k.parameters():
    param.requires_grad = False
for param in model.final_row.parameters():
    param.requires_grad = False
for param in model.final_col.parameters():
    param.requires_grad = False

# Collect parameters not belonging to the AFA modules.
non_k_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(non_k_params, lr=BACKBONE_LR)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

print("=== Stage 1: Training CNN and GM Network Only (using ground truth kgt) ===")
for epoch in range(stage1_epochs):
    model.train()
    epoch_loss_sum = 0.0
    num_iterations = 25  # fixed number of iterations per epoch
    for iter_num in range(num_iterations):
        optimizer.zero_grad()
        # In stage 1, ground truth k (kgt) is used as input (inside the model forward)
        outputs = model(single_sample, regression=True)  # model uses gt k in training mode
        loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.item()
        if iter_num % 5 == 0:
            print(f"Stage 1 - Epoch {epoch}, Iteration {iter_num}, Loss: {loss.item():.4f}")
    epoch_loss = epoch_loss_sum / num_iterations
    print(f"Stage 1 - Epoch {epoch} average loss: {epoch_loss:.4f}")
    # Save checkpoint for stage 1
    save_model(model, str(checkpoint_path / f'stage1_params_{epoch+1:04}.pt'))
    torch.save(optimizer.state_dict(), str(checkpoint_path / f'stage1_optim_{epoch+1:04}.pt'))
    scheduler.step()

# -------------------------------
# Stage 2: Warm-up Training for AFA Modules Only.
# Freeze CNN and GM network; only train AFA modules.
print("=== Stage 2: Training AFA Modules Only (Warm-up) ===")
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False
# Unfreeze only AFA modules
for param in model.encoder_k.parameters():
    param.requires_grad = True
for param in model.final_row.parameters():
    param.requires_grad = True
for param in model.final_col.parameters():
    param.requires_grad = True

# Prepare optimizer for AFA modules
k_params = list(model.encoder_k.parameters()) + list(model.final_row.parameters()) + list(model.final_col.parameters())
optimizer_k = optim.Adam(k_params, lr=K_LR)
scheduler_k = optim.lr_scheduler.MultiStepLR(optimizer_k, milestones=[3], gamma=0.1)

for epoch in range(stage2_epochs):
    model.train()
    epoch_loss_sum = 0.0
    num_iterations = 25
    for iter_num in range(num_iterations):
        optimizer_k.zero_grad()
        # In stage 2, still use ground truth k as input (to provide stable supervision to AFA)
        outputs = model(single_sample, regression=True)
        loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
        # Also include AFA (k) loss from the regression branch
        ks_loss = outputs.get('ks_loss', 0.)
        total_loss = loss + ks_loss
        total_loss.backward()
        optimizer_k.step()
        epoch_loss_sum += total_loss.item()
        if iter_num % 5 == 0:
            print(f"Stage 2 - Epoch {epoch}, Iteration {iter_num}, Loss: {total_loss.item():.4f}")
    epoch_loss = epoch_loss_sum / num_iterations
    print(f"Stage 2 - Epoch {epoch} average loss: {epoch_loss:.4f}")
    # Save checkpoint for stage 2
    save_model(model, str(checkpoint_path / f'stage2_params_{epoch+1:04}.pt'))
    torch.save(optimizer_k.state_dict(), str(checkpoint_path / f'stage2_optim_{epoch+1:04}.pt'))
    scheduler_k.step()

# -------------------------------
# Stage 3: Joint Training of All Modules.
# Unfreeze all parameters.
print("=== Stage 3: Joint Training of CNN, GM network, and AFA Modules ===")
for param in model.parameters():
    param.requires_grad = True
# Use a joint optimizer for all parameters.
optimizer_joint = optim.Adam(model.parameters(), lr=LR)
scheduler_joint = optim.lr_scheduler.MultiStepLR(optimizer_joint, milestones=[5, 10], gamma=0.1)

for epoch in range(stage3_epochs):
    model.train()
    epoch_loss_sum = 0.0
    num_iterations = 25
    for iter_num in range(num_iterations):
        optimizer_joint.zero_grad()
        # In stage 3, use predicted k (k_pred) from AFA modules.
        # The model's forward should then use its internal k prediction.
        outputs = model(single_sample, regression=True)
        loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
        ks_loss = outputs.get('ks_loss', 0.)
        total_loss = loss + ks_loss
        total_loss.backward()
        optimizer_joint.step()
        epoch_loss_sum += total_loss.item()
        if iter_num % 5 == 0:
            print(f"Stage 3 - Epoch {epoch}, Iteration {iter_num}, Loss: {total_loss.item():.4f}")
    epoch_loss = epoch_loss_sum / num_iterations
    print(f"Stage 3 - Epoch {epoch} average loss: {epoch_loss:.4f}")
    # Save checkpoint for stage 3
    save_model(model, str(checkpoint_path / f'stage3_params_{epoch+1:04}.pt'))
    torch.save(optimizer_joint.state_dict(), str(checkpoint_path / f'stage3_optim_{epoch+1:04}.pt'))
    scheduler_joint.step()

# -------------------------------
# Early Stopping and Final Model Saving (Optional)
# -------------------------------
patience = 3
best_loss = float('inf')
no_improvement_count = 0

# Here, you could integrate your early stopping logic over a validation dataset.
# TODO: Add validation loop to compute validation loss and update early stopping logic.

# -------------------------------
# Final Evaluation on a Sample
# -------------------------------
best_model_path = str(checkpoint_path / 'best_model.pt')
print("Loading the best model for evaluation...")
load_model(model, best_model_path, strict=False)
model.eval()
with torch.no_grad():
    outputs = model(single_sample, regression=False)

# Assuming outputs['ds_mat'] is the similarity matrix between keypoints
ds_mat = outputs['ds_mat'].cpu().numpy().squeeze(0)
per_mat = outputs['perm_mat'].cpu().numpy().squeeze(0)

# Retrieve keypoints (or use dummy placeholders if not available)
if 'Ps' in single_sample:
    kp0 = single_sample['Ps'][0].cpu().numpy().squeeze(0)
    kp1 = single_sample['Ps'][1].cpu().numpy().squeeze(0)
else:
    kp0 = np.array([[100, 100], [150, 150], [200, 200]])
    kp1 = np.array([[110, 110], [160, 160], [210, 210]])

# Use OpenCV DMatch and draw keypoints
cv2_kp0 = [cv2.KeyPoint(x=float(k[0]), y=float(k[1]), size=1) for k in kp0]
cv2_kp1 = [cv2.KeyPoint(x=float(k[0]), y=float(k[1]), size=1) for k in kp1]

matches = build_matches(ds_mat, per_mat)

if 'img0_path' in single_sample and 'img1_path' in single_sample:
    img0 = cv2.imread(single_sample['img0_path'])
    img1 = cv2.imread(single_sample['img1_path'])
else:
    img0 = cv2.imread('/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg')
    img1 = cv2.imread('/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg')
    print("Using fallback image paths.")

img0_kp = cv2.drawKeypoints(img0, cv2_kp0, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kp = cv2.drawKeypoints(img1, cv2_kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("image0_keypoints.jpg", img0_kp)
cv2.imwrite("image1_keypoints.jpg", img1_kp)

img_matches = cv2.drawMatches(img0, cv2_kp0, img1, cv2_kp1, matches, None, flags=2)
cv2.imwrite("matching_result.jpg", img_matches)
print("Matching result saved as 'matching_result.jpg'.")
