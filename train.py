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
from torch.utils.tensorboard import SummaryWriter


from src.train.data_loader import build_dataloaders
from src.train.training_loop import train_epoch
from src.train.evaluation import validate_epoch, test_evaluation
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from src.parallel import DataParallel
from src.loss_func import PermutationLoss
from utils.models_sl import save_model, load_model
from utils.visualize import visualize_stochastic_matrix, visualize_match, to_grayscale_cv2_image
from src.evaluation_metric import matching_accuracy
from utils.scheduler import WarmupScheduler
# Utility function for generating cv2.DMatch lists
from utils.matching import build_matches
# from apex import amp



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

start_epoch = float('inf')
# config_files = ["stage1.yml", "stage2.yml", "stage3.yml", "stage4.yml", "stage5.yml"]
# config_files = ["stage1.yml", "stage2.yml", "stage3.yml"]
config_files = ["stage4.yml" ]
start_path = Path("checkpoints")
start_path.mkdir(parents=True, exist_ok=True)
start_file = start_path / "checkpoint.json"

# Create TensorBoard log directory
log_dir = Path("logs/tensorboard")
log_dir.mkdir(parents=True, exist_ok=True)

PRETRAINED_PATH = "results/base/params/best_model.pt" 


for file in config_files:
    scheduler = scheduler_k = None
    print("Using config ", file)
    # ====================================================
    # Load Settings from YAML Configuration File
    # =====================================================
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]

    # Create a new writer for this training stage
    writer = SummaryWriter(log_dir=str(log_dir / file.split('.')[0]))
    
    # Hyperparameters from config
    if os.path.exists(start_file):
        with open(start_file, "r") as f:
            start_data = json.load(f)
            start_epoch = start_data.get("start_epoch", 0)  # Default to 0 if not found
            print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = train_config.get("start_epoch", 0)
    num_iterations = train_config.get("num_iterations", 25)
    # The K_Optimize flag from config will be overridden based on stage below.
    BATCH_SIZE     = train_config.get("BATCH_SIZE", 1)
    LR             = train_config.get("LR", 2e-3)
    BACKBONE_LR    = train_config.get("BACKBONE_LR", 2e-5)
    print("BACKBONE_LR =", BACKBONE_LR)
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
    train_root = 'dataset/Synthetic'
    OUTPUT_PATH = "results/binary-classifier"

    # =====================================================
    # Setup Logging
    # =====================================================
    logging.basicConfig(
        filename='fp.log', 
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)

    # -----------------------------------------------------
    # Determine training stage from config filename
    # -----------------------------------------------------
    if "stage1" in file:
        stage = 1
    elif "stage2" in file:
        stage = 2
    elif "stage3" in file:
        stage = 3
    elif "stage4" in file:
        stage = 4
    elif "stage5" in file:
        stage = 5
    else:
        stage = None

    # =====================================================
    # Dataset and Dataloader
    # =====================================================
    task = 'classify' if stage in (4, 5) else 'match'
    dataloader, val_dataloader, test_dataloader = build_dataloaders(train_root, dataset_len, task=task)
    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(regression=REGRESSION)
    criterion = PermutationLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Uncomment below if using multiple GPUs:
    # model = DataParallel(model, device_ids=[0,1])
    # model, optimizer = amp.initialize(model, optimizer)
    # =====================================================
    # Freeze / Unfreeze Layers Based on Stage
    # =====================================================
    # Set up Optimizers (with optional separate k_optimizer)
    # =====================================================
    # For stage 1, we update only non-k_params; for stage 2, only k_params are trainable.
    # In stage 3, we use the original parameter grouping.
    backbone_ids = [id(item) for item in model.backbone_params]
    k_params = model.k_params_id
    other_params = [param for param in model.parameters() if id(param) not in k_params and id(param) not in backbone_ids]
    model_params = [
        {'params': other_params},
        {'params': model.backbone_params, 'lr': BACKBONE_LR}
        ]
    match_cls_ids = [id(p) for p in model.match_cls.parameters()]
    if stage == 1:
        
        print("Stage 1: Freezing all layers in k_params and training other parameters.")
        # Freeze k_params
        for param in model.k_params:
            for p in param["params"]:
                p.requires_grad = False
        # # Ensure all other parameters are trainable
       # In stage 1 we do not optimize the k_params
        K_Optimize = False

        # Only parameters with requires_grad == True will be optimized.
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = None
    elif stage == 2:
    
        print("Stage 2: Freezing all parameters except k_params (which are unfrozen).")
        for name, param in model.named_parameters():
            if id(param) not in model.k_params_id:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # In stage 2, we optimize only the k_params.
        K_Optimize = True
      
        # In stage 2, only k_params are trainable.
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)
    elif stage == 3:
        # Stage 3: All parameters are trainable. We separate backbone parameters for a different LR.s
        
        
        print("Stage 3: Unfreezing all layers for full fine-tuning.")
        # Unfreeze every parameter
        for name,  param in model.named_parameters():
            param.requires_grad = True
        K_Optimize = True


        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-4)
    elif stage == 4:
        print("Stage 4: Classification training, optimizing only k parameters.")
        for name, param in model.named_parameters():
            if id(param) not in model.k_params_id:
                param.requires_grad = False
            else:
                param.requires_grad = True
        K_Optimize = True
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)
    elif stage == 5:
        print("Stage 5: Training match classifier only.")
        for name, param in model.named_parameters():
            param.requires_grad = id(param) in match_cls_ids
        K_Optimize = False
        optimizer = optim.AdamW(model.match_cls.parameters(), lr=LR, weight_decay=1e-4)
        optimizer_k = None
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        optimizer_k = None

    # =====================================================
    # Schedulers for Both Optimizers
    # =====================================================
    # milestones = [int(0.6 * num_epochs), int(0.7 * num_epochs), int(0.9 * num_epochs)]
    # milestones = [int(0.025*num_epochs),int(0.05*num_epochs),int(0.2*num_epochs),int(0.4*num_epochs),int(0.6*num_epochs),int(0.8*num_epochs),int(0.1*num_epochs)]
    warmup_epochs = 10
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            #    milestones=milestones,
                                            #    gamma=LR_DECAY,
                                            #    last_epoch=-1)
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=LR_DECAY)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)
    if optimizer_k is not None:
        # main_scheduler_k = optim.lr_scheduler.MultiStepLR(optimizer_k,
        #                                               milestones=milestones,
        #                                               gamma=LR_DECAY,
        #                                               last_epoch=-1)
        main_scheduler_k = optim.lr_scheduler.ReduceLROnPlateau(optimizer_k, patience=1, factor=LR_DECAY)
        scheduler_k = WarmupScheduler(optimizer_k, warmup_epochs=2, after_scheduler=main_scheduler_k)

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
        load_model(model, model_path)
    if len(optim_path) > 0:
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))
    if len(optim_k_path) > 0:
        try:
            print("Loading optimizer_k state from {}".format(optim_k_path))
            optimizer_k.load_state_dict(torch.load(optim_k_path))
        except FileNotFoundError:
            print("No optimizer_k checkpoint found; starting fresh for k_params.")
    PRETRAINED_PATH = ""  # Clear after loading to avoid reloading in next stages
    # Initialize warmup scheduler learning rates after loading optimizer state
    # if start_epoch == 0:  # Only for fresh training, not resuming
    print("Initializing warmup learning rates for first epoch...")
    # Set the initial warmup learning rates
    initial_lr = scheduler.get_initial_lr() if hasattr(scheduler, 'get_initial_lr') else LR / warmup_epochs
    initial_backbone_lr = BACKBONE_LR / warmup_epochs  # Apply warmup to backbone LR too
    
    for param_group in optimizer.param_groups:
        if param_group == optimizer.param_groups[-1]:  # Backbone group (assuming it's last)
            param_group['lr'] = initial_backbone_lr
        else:  # Other parameter groups
            param_group['lr'] = initial_lr
    
    if optimizer_k is not None and scheduler_k is not None:
        initial_k_lr = scheduler_k.get_initial_lr() if hasattr(scheduler_k, 'get_initial_lr') else K_LR / warmup_epochs
        for param_group in optimizer_k.param_groups:
            param_group['lr'] = initial_k_lr
            
    
    # best_model_path = str(checkpoint_path / "best_model.pt")
    # if os.path.exists(best_model_path):
    #     print(f"Loading best model weights from {best_model_path} before training loop...")
    #     load_model(model, best_model_path)
                
    

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

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)

        if optimizer_k is not None:
            for i, param_group in enumerate(optimizer_k.param_groups):
                writer.add_scalar(f'Learning_Rate_K/group_{i}', param_group['lr'], epoch)

        avg_epoch_loss, avg_ks_loss, avg_total_loss, avg_accuracy = train_epoch(
            model,
            dataloader,
            criterion,
            optimizer,
            optimizer_k,
            device,
            writer,
            epoch,
            start_epoch,
            stage,
            logger,
            checkpoint_path,
        )
            
        # =====================================================
        # ---- Validation after each epoch ----
        # =====================================================
        avg_val_loss, avg_ks_loss, avg_val_total, avg_val_accuracy = validate_epoch(
            model,
            val_dataloader,
            criterion,
            device,
            writer,
            epoch,
            logger,
        )
    
        
        # Save best model based on validation loss and update checkpoint file
        if avg_val_total < best_loss:
            best_loss = avg_val_total
            no_improvement_count = 0
            best_model_path = str(checkpoint_path / "best_model.pt")
            save_model(model, best_model_path)
            with open(start_file, "w") as f:
                json.dump({"start_epoch": epoch + 1}, f)
        else:
            no_improvement_count += 1
            print("No improvement for {} epoch(s). Best loss so far: {:.4f}".format(no_improvement_count, best_loss))
            if no_improvement_count >= patience:
                print("Stopping early at epoch {} due to no improvement.".format(epoch + 1))
                break

        

        # Initialize previous learning rates on first epoch
        if epoch == start_epoch:
            prev_lr = [group['lr'] for group in optimizer.param_groups]
            if optimizer_k is not None:
                prev_k_lr = [group['lr'] for group in optimizer_k.param_groups]

        # Step LR schedulers
        scheduler.step(avg_val_loss)
        if optimizer_k is not None:
            scheduler_k.step(avg_ks_loss)

        # Detect LR reduction for main optimizer
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        lr_reduced = any(clr < plr for clr, plr in zip(curr_lr, prev_lr))
        prev_lr = curr_lr  # Update previous for next iteration

        if lr_reduced:
            print("[LR REDUCED] Reloading best model weights from", checkpoint_path / "best_model.pt")
            best_model_path = str(checkpoint_path / "best_model.pt")
            load_model(model, best_model_path)
            
       
        
        # ---- Test Evaluation Periodically ----
        if epoch % 5 == 0:
            test_evaluation(
                model,
                test_dataloader,
                criterion,
                device,
                writer,
                epoch,
                stage,
            )
    
    # Close the TensorBoard writer at the end of this training stage
    writer.close()

# =====================================================
# Load Best Model and Evaluate on a Sample
# =====================================================
single_sample = next(iter(val_dataloader))
single_sample = data_to_cuda(single_sample)
print(single_sample.keys())

# os.remove(start_file)
best_model_path = str(checkpoint_path / "best_model.pt")
print("Loading the best model for evaluation...")
load_model(model, best_model_path)
model.eval()
with torch.no_grad():
    outputs = model(single_sample)
    
acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'], idx=0)
if isinstance(acc, torch.Tensor):
    if acc.numel() > 1:  # Check if tensor has multiple elements
        acc = acc.mean().item()  # Take the mean before converting to scalar
    else:
        acc = acc.item()

                
    
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

# print("ds_mat shape:", ds_mat.shape)
# print(ds_mat)
# print("per_mat shape:", per_mat.shape)
# print(per_mat)
# visualize_stochastic_matrix(per_mat, "Perm_matrix")
# print("Number of keypoints in image0 (from kp0):", kp0.shape[0])
# print("Number of keypoints in image1 (from kp1):", kp1.shape[0])

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
print("Accuracy: ", acc)

# Add final visualizations to TensorBoard
if len(matches) > 0:
    match_path = f"photos/final_match-train.jpg"
    visualize_match(img0, img1, kp0, kp1, matches, prefix="photos/", filename="final_match-train")
    visualize_stochastic_matrix(ds_mat, filename="matrix-train")

    if os.path.exists(match_path):
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        writer.add_image('Final/Matches', match_img.transpose(2, 0, 1), dataformats='CHW')

writer.add_scalar('Final/Accuracy', acc, 0)
writer.close()

print("Accuracy: ", acc)
