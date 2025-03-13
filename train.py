#!/usr/bin/env python
import logging
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch
import torch.optim as optim
import json
import os

from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader
from ngm import Net
from utils.data_to_cuda import data_to_cuda
from src.parallel import DataParallel
from src.loss_func import PermutationLoss
from utils.models_sl import save_model, load_model
from utils.visualize import visualize_stochastic_matrix

start_epoch=float('inf')
config_files = ["stage1.yml", "stage2.yml", "stage3.yml"]
start_path = Path("checkpoints")
start_path.mkdir(parents=True, exist_ok=True)
start_file = start_path / "checkpoint.json"

for file in  config_files:
    print("Using config ", file)
# ====================================================
# Load Settings from YAML Configuration File
# =====================================================
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]

    # Hyperparameters from config
    if os.path.exists(start_file):
        with open(start_file, "r") as f:
            start_data = json.load(f)
            start_epoch = start_data.get("start_epoch", 0)  # Default to 0 if not found
            print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch    = train_config.get("start_epoch", 0)
    num_iterations = train_config.get("num_iterations", 25)
    K_Optimize     = train_config.get("K_Optimize", False)
    BATCH_SIZE     = train_config.get("BATCH_SIZE", 1)
    LR             = train_config.get("LR", 2e-3)
    BACKBONE_LR    = train_config.get("BACKBONE_LR", 2e-5)
    print(BACKBONE_LR)
    K_LR           = train_config.get("K_LR", 2e-3)
    LR_DECAY       = train_config.get("LR_DECAY", 0.5)
    patience       = train_config.get("patience", 10)
    num_epochs     = train_config.get("num_epochs", 10)
    
    ngm_config = config.get("ngm", {})

    REGRESSION    = ngm_config.get("REGRESSION", True)
    
    print("Start epoch: ", start_epoch)
    # =====================================================
    # Hard-Coded and Derived Parameters
    # =====================================================
    dataset_len = 640

    best_loss = float('inf')
    no_improvement_count = 0

    # File paths
    train_root = '/green/data/L3SF_V2/L3SF_V2_Augmented'
    OUTPUT_PATH = "result"
    PRETRAINED_PATH = ""  # Set this to a pretrained model path if needed

    # =====================================================
    # Setup Logging
    # =====================================================
    logging.basicConfig(
    filename='fp.log', 
    level=logging.DEBUG
)
    logger = logging.getLogger(__name__)

    # =====================================================
    # Dataset and Dataloader
    # =====================================================
    benchmark = L3SFV2AugmentedBenchmark(
        sets='train',
        obj_resize=(320, 240),
        train_root=train_root
    )

    image_dataset = GMDataset("L3SFV2Augmented", benchmark, dataset_len, True, None, "2GM")
    dataloader = get_dataloader(image_dataset, shuffle=True, fix_seed=True)

    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(regression=REGRESSION)
    criterion = PermutationLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Uncomment below if using multiple GPUs:
    # model = DataParallel(model, device_ids=[0,1])

    # =====================================================
    # Set up Optimizers (with optional separate k_optimizer)
    # =====================================================
    backbone_ids = [id(item) for item in model.backbone_params]
    if K_Optimize:
        k_params = model.k_params_id
        other_params = [param for param in model.parameters() if id(param) not in k_params and id(param) not in backbone_ids]
        model_params = [
            {'params': other_params},
            {'params': model.backbone_params, 'lr': BACKBONE_LR}
        ]
    else:
        other_params = [param for param in model.parameters() if id(param) not in backbone_ids]
        model_params = [
            {'params': other_params},
            {'params': model.backbone_params, 'lr': BACKBONE_LR}
        ]
    optimizer = optim.Adam(model_params, lr=LR)
    optimizer_k = optim.Adam(model.k_params, lr=K_LR) if K_Optimize else None

    # =====================================================
    # Schedulers for Both Optimizers
    # =====================================================
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[10, 20, 30],
                                                gamma=LR_DECAY,
                                                last_epoch=-1)
    if optimizer_k is not None:
        scheduler_k = optim.lr_scheduler.MultiStepLR(optimizer_k,
                                                    milestones=[10, 20, 30],
                                                    gamma=LR_DECAY,
                                                    last_epoch=-1)

    # =====================================================
    # Checkpoint Loading (if start_epoch > 0)
    # =====================================================
    checkpoint_path = Path(OUTPUT_PATH) / 'params'
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_path = ""
    optim_path = ""
    optim_k_path = ""
    if start_epoch != 0:
        model_path = str(checkpoint_path / f'params_{start_epoch:04}.pt')
        optim_path = str(checkpoint_path / f'optim_{start_epoch:04}.pt')
        if optimizer_k is not None:
            optim_k_path = str(checkpoint_path / f'optim_k_{start_epoch:04}.pt')

    if len(PRETRAINED_PATH) > 0:
        model_path = PRETRAINED_PATH

    if len(model_path) > 0:
        print("Loading model parameters from {}".format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))
    if len(optim_k_path) > 0:
        try:
            print("Loading optimizer_k state from {}".format(optim_k_path))
            optimizer_k.load_state_dict(torch.load(optim_k_path))
        except FileNotFoundError:
            print("No optimizer_k checkpoint found; starting fresh for AFA modules.")



    # =====================================================
    # Training Loop
    # =====================================================
    for epoch in range(start_epoch, start_epoch + num_epochs):
        logger.info(f"Epoch {epoch}/{start_epoch + num_epochs - 1}")
        logger.info("-" * 50)
        print("Epoch {}/{}".format(epoch, start_epoch + num_epochs - 1))
        print("-" * 10)
        
        model.train()
        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print("K_regression_lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer_k.param_groups]))
        
        epoch_loss_sum = 0.0
        running_ks_loss = 0.0
        running_ks_error = 0.0
        
        for iter_num in range(num_iterations):
            # =====================================================
            # Fetch a Single Sample (for repeated training iterations)
            # =====================================================
            single_sample = next(iter(dataloader))
            single_sample = data_to_cuda(single_sample)
            # print(single_sample["id_list"])
            # print(single_sample.keys())
            # print("Kpts:", single_sample["ns"])
            
            
            
            
            optimizer.zero_grad()
            if optimizer_k is not None:
                optimizer_k.zero_grad()
            
            outputs = model(single_sample)
            loss = criterion(outputs["ds_mat"], outputs["gt_perm_mat"], *outputs["ns"])
            
            # Combine losses in one backward pass
            if "ks_loss" in outputs and K_Optimize:
                ks_loss = outputs["ks_loss"]
                running_ks_loss += ks_loss.item()
                running_ks_error += outputs.get("ks_error", 0).item() if "ks_error" in outputs else 0.0
                loss.backward()
                ks_loss.backward()
                optimizer.step()
                optimizer_k.step()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss_sum += loss.item()
            
         
            if iter_num % 5 == 0:
                if "ks_loss" in outputs and K_Optimize:
                    print("Epoch: {}, Iteration: {}, Loss: {:.4f}, ks_loss: {:.4f}".format(
                        epoch, iter_num, loss.item(), ks_loss.item()))
                    logger.info(f"Epoch: {epoch}, Iteration: {iter_num}, Loss: {loss.item():.4f}, ks_loss: {ks_loss.item():.4f}")
                    
                else:
                    print("Epoch: {}, Iteration: {}, Loss: {:.4f}".format(
                        epoch, iter_num, loss.item()))
                    logger.info(f"Epoch: {epoch}, Iteration: {iter_num}, Loss: {loss.item():.4f}")
                    
        avg_epoch_loss = epoch_loss_sum / num_iterations
            
            
        print("==> End of Epoch {}, average loss = {:.4f}, total ks_loss = {:.4f}".format(
            epoch, avg_epoch_loss, running_ks_loss/num_iterations))
        logger.info(f"==> End of Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}, Total ks_loss: {running_ks_loss/num_iterations:.4f}")

        # Save checkpoints for model, optimizer, and optimizer_k
        save_model(model, str(checkpoint_path / f"params_{epoch + 1:04}.pt"))
        torch.save(optimizer.state_dict(), str(checkpoint_path / f"optim_{epoch + 1:04}.pt"))
        if optimizer_k is not None:
            torch.save(optimizer_k.state_dict(), str(checkpoint_path / f"optim_k_{epoch + 1:04}.pt"))
        
        # Step LR schedulers
        scheduler.step()
        if optimizer_k is not None:
            scheduler_k.step()
            
    
        # EARLY STOPPING based on average loss (lower is better)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement_count = 0
            with open(start_file, "w") as f:
                json.dump({"start_epoch": epoch + 1}, f)
            best_model_path = str(checkpoint_path / "best_model.pt")
            save_model(model, best_model_path)
        else:
            no_improvement_count += 1
            print("No improvement for {} epoch(s). Best loss so far: {:.4f}".format(
                no_improvement_count, best_loss))
            if no_improvement_count >= patience:
                print("Stopping early at epoch {} due to no improvement for {} epochs.".format(
                    epoch + 1, patience))
                
                break

# =====================================================
# Load Best Model and Evaluate on a Sample
# =====================================================
single_sample = next(iter(dataloader))
single_sample = data_to_cuda(single_sample)
print(single_sample.keys())

# os.remove(start_file)
best_model_path = str(checkpoint_path / "best_model.pt")
print("Loading the best model for evaluation...")
load_model(model, best_model_path, strict=False)
model.eval()
with torch.no_grad():
    outputs = model(single_sample)
    
# Explicitly select the first sample from the batch
kp0_tensor = single_sample['Ps'][0][0]  # shape: [num_keypoints_img0, 2]
kp0 = kp0 = kp1 = None

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

print("ds_mat shape:", ds_mat.shape)
print(ds_mat)
print("per_mat shape:", per_mat.shape)
print(per_mat)
visualize_stochastic_matrix(per_mat, "Perm_matrix")
print("Number of keypoints in image0 (from kp0):", kp0.shape[0])
print("Number of keypoints in image1 (from kp1):", kp1.shape[0])

matches = []
for i in range(ds_mat.shape[0]):
    valid_indices = np.where(per_mat[i] == 1)[0]
    if valid_indices.size == 0:
        continue
    best_index = valid_indices[np.argmax(ds_mat[i, valid_indices])]
    distance_value = np.squeeze(ds_mat[i, best_index])
    if hasattr(distance_value, "size") and distance_value.size != 1:
        distance_value = distance_value.flatten()[0]
    distance = float(distance_value)
    matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_index, _imgIdx=0, _distance=distance))

if "id_list" in single_sample:
    img0 = cv2.imread(benchmark.get_path(single_sample["id_list"][0][0]))
    img1 = cv2.imread(benchmark.get_path(single_sample["id_list"][1][0]))
else:
    img0 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_0.jpg")
    img1 = cv2.imread("/green/data/L3SF_V2/L3SF_V2_Augmented/R1/8_right_loop_aug_1.jpg")
    print("Using fallback image paths.")

cv2_kp0 = []
for kp in kp0:
    x = float(np.array(kp[0]).flatten()[0])
    y = float(np.array(kp[1]).flatten()[0])
    cv2_kp0.append(cv2.KeyPoint(x=x, y=y, size=1))
    
cv2_kp1 = []
for kp in kp1:
    x = float(np.array(kp[0]).flatten()[0])
    y = float(np.array(kp[1]).flatten()[0])
    cv2_kp1.append(cv2.KeyPoint(x=x, y=y, size=1))
    
img0_kp = cv2.drawKeypoints(img0, cv2_kp0, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img1_kp = cv2.drawKeypoints(img1, cv2_kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("image0_keypoints.jpg", img0_kp)
cv2.imwrite("image1_keypoints.jpg", img1_kp)

print("cv2_kp0 length:", len(cv2_kp0))
print("cv2_kp1 length:", len(cv2_kp1))
print("Number of matches found:", len(matches))

img_matches = cv2.drawMatches(img0, cv2_kp0, img1, cv2_kp1, matches, None, flags=2)
cv2.imwrite("matching_result.jpg", img_matches)
print("Matching result saved as 'matching_result.jpg'.")
