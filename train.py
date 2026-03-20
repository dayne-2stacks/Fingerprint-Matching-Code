#!/usr/bin/env python
import argparse
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
from torch.nn.modules.batchnorm import _BatchNorm


from src.train.data_loader import build_dataloaders
from src.train.training_loop import train_epoch
from src.train.evaluation import validate_epoch, test_evaluation
from src.model.ngm2 import (
    Net,
)
from utils.data_to_cuda import data_to_cuda
from src.parallel import DataParallel
from src.loss_func import WeightedPermLoss, PermutationLoss, PermutationLossHung, FocalLoss
from utils.models_sl import save_model, load_model, load_optimizer
from utils.visualize import visualize_stochastic_matrix, visualize_match, to_grayscale_cv2_image
from src.evaluation_metric import matching_accuracy
from utils.scheduler import WarmupScheduler
from src.model.dustbin import ns_pair_to_ints, strip_dustbin_from_outputs
# Utility function for generating cv2.DMatch lists
from utils.matching import build_matches
from utils.setup import (
    _load_global_config,
    _load_stage_config,
    _collect_stage_output_paths,
    _stage_from_filename,
    _stage_group_label,
    _configure_stage_trainability,
    _build_optimizers,
    _default_pretrained_path,
)
# from apex import amp


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def _set_batchnorm_eval(module):
    """Set BatchNorm layers to eval mode so running stats stay frozen."""
    if isinstance(module, _BatchNorm):
        module.eval()


def _parse_args():
    parser = argparse.ArgumentParser(description="Train fingerprint matching model")
    parser.add_argument(
        "--exp-name", default="",
        help="Experiment name — namespaces all outputs under runs/<exp-name>/",
    )
    parser.add_argument(
        "--config-dir", default="config2",
        help="Directory containing stage*.yml files (and optionally settings.yml). Default: config2",
    )
    parser.add_argument(
        "--global-config", default="",
        help="Path to global settings YAML. Defaults to <config-dir>/settings.yml if present, else settings.yml",
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override any config value, e.g. --set LR=0.001 BATCH_SIZE=8",
    )
    return parser.parse_args()

args = _parse_args()
start_epoch = float('inf')

# Discover stage configs from --config-dir (sorted so stage1 < stage2 < ...)
_config_dir = Path(args.config_dir)
STAGE_CONFIG_FILES = sorted(str(p) for p in _config_dir.glob("stage*.yml"))
if not STAGE_CONFIG_FILES:
    raise FileNotFoundError(f"No stage*.yml files found in {_config_dir}")

# Resolve global config: explicit flag > <config-dir>/settings.yml > settings.yml
if args.global_config:
    GLOBAL_CONFIG_FILE = args.global_config
elif (_config_dir / "settings.yml").exists():
    GLOBAL_CONFIG_FILE = str(_config_dir / "settings.yml")
else:
    GLOBAL_CONFIG_FILE = "settings.yml"

print(f"Config dir:    {_config_dir}")
print(f"Global config: {GLOBAL_CONFIG_FILE}")
print(f"Stage configs: {STAGE_CONFIG_FILES}")


global_cfg = _load_global_config(GLOBAL_CONFIG_FILE)
config_files = list(STAGE_CONFIG_FILES)
stage_configs = {cfg_file: _load_stage_config(cfg_file, global_cfg) for cfg_file in config_files}

# Namespace all outputs under runs/<exp-name>/ when --exp-name is given
if args.exp_name:
    exp_root = Path("runs") / args.exp_name
    for cfg in stage_configs.values():
        cfg["OUTPUT_PATH"]    = str(exp_root / cfg["OUTPUT_PATH"])
        cfg["CHECKPOINT_PATH"] = str(exp_root / cfg["CHECKPOINT_PATH"])
        cfg["LOG_DIR"]        = str(exp_root / cfg["LOG_DIR"])

# Apply --set KEY=VALUE overrides to every stage config
overrides = {}
for item in (args.set or []):
    k, v = item.split("=", 1)
    overrides[k] = yaml.safe_load(v)  # handles ints, floats, bools
if overrides:
    for cfg in stage_configs.values():
        for k, v in overrides.items():
            if k in cfg:
                cfg[k] = v

stage_output_paths = _collect_stage_output_paths(config_files, stage_configs)
global_defaults_log = {
    "config_file": GLOBAL_CONFIG_FILE,
    "train_defaults": global_cfg["train_defaults"],
    "policy_defaults": global_cfg["policy_defaults"],
    "data_defaults": global_cfg["data_defaults"],
}
print(f"Loaded global defaults: {global_defaults_log}")


for file in config_files:
    scheduler = scheduler_k = None
    print("Using config ", file)

    stage_cfg = stage_configs[file]
    stage = _stage_from_filename(file)
    stage_group = _stage_group_label(stage)
    stage_name = f"stage{stage}"

    OUTPUT_PATH = stage_cfg["OUTPUT_PATH"]
    stage_output_paths[int(stage)] = OUTPUT_PATH
    PRETRAINED_PATH = stage_cfg["PRETRAINED_PATH"]
    CHECKPOINT_PATH = stage_cfg["CHECKPOINT_PATH"]
    checkpoint_root = Path(CHECKPOINT_PATH) / stage_group / stage_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    start_file = checkpoint_root / "checkpoint.json"
    log_dir = Path(stage_cfg["LOG_DIR"]) / stage_group / stage_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(OUTPUT_PATH) / "params"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create a new writer for this training stage
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Hyperparameters from config
    if os.path.exists(start_file):
        with open(start_file, "r") as f:
            start_data = json.load(f)
            start_epoch = start_data.get("start_epoch", 0)  # Default to 0 if not found
            print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = int(stage_cfg["start_epoch"])

    num_iterations = int(stage_cfg["num_iterations"])
    num_epochs = int(stage_cfg["num_epochs"])
    BATCH_SIZE = int(stage_cfg["BATCH_SIZE"])
    BM_NAME = stage_cfg["BM_NAME"]
    FILTER = stage_cfg["FILTER"]
    OVERFIT_TO_TRAIN_SPLIT = bool(stage_cfg["OVERFIT_TO_TRAIN_SPLIT"])
    LR = float(stage_cfg["LR"])
    BACKBONE_LR = float(stage_cfg["BACKBONE_LR"])
    K_LR = float(stage_cfg["K_LR"])
    LR_DECAY = float(stage_cfg["LR_DECAY"])
    patience = int(stage_cfg["patience"])

    WARMUP_EPOCHS = int(stage_cfg["WARMUP_EPOCHS"])
    WARMUP_K_EPOCHS = int(stage_cfg["WARMUP_K_EPOCHS"])

    REGRESSION = bool(stage_cfg["REGRESSION"])
    DUSTBIN_REJECT_ENABLE = bool(stage_cfg["DUSTBIN_REJECT_ENABLE"])
    DETECT_ANOMALY = bool(stage_cfg["DETECT_ANOMALY"])
    FOCAL_GAMMA = float(stage_cfg.get("FOCAL_GAMMA", 1.0))


    print("BACKBONE_LR =", BACKBONE_LR)
    print("Start epoch: ", start_epoch)


    
    # =====================================================
    # Hard-Coded and Derived Parameters
    # =====================================================
    dataset_len = int(global_cfg["data_defaults"]["dataset_len"])

    best_loss = float('inf')
    no_improvement_count = 0

    # File paths
    # Default to the synthetic dataset; override for stage 6
    train_root = str(global_cfg["data_defaults"]["train_root"])
    # OUTPUT_PATH = "results/base"

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
    dataloader, val_dataloader, test_dataloader = build_dataloaders(
        train_root,
        dataset_len,
        batch_size=BATCH_SIZE,
        benchmark_name=BM_NAME,
        filter=FILTER,
        overfit_to_train_split=OVERFIT_TO_TRAIN_SPLIT,
        stage=stage,
        has_dustbin=DUSTBIN_REJECT_ENABLE,
    )
    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(has_dustbin=DUSTBIN_REJECT_ENABLE)

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    # criterion = PermutationLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Uncomment below if using multiple GPUs:
    # model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # model, optimizer = amp.initialize(model, optimizer)

    # =====================================================
    # Stage-wise progressive unfreezing
    # =====================================================
    stage_param_groups = _configure_stage_trainability(model, stage)
    optimizer, optimizer_k = _build_optimizers(model, stage_param_groups, LR, BACKBONE_LR, K_LR)

    stage_messages = {
        1: "Stage 1: train shared matcher baseline (matcher/backbone active; k/dustbin frozen).",
        2: "Stage 2: train K heads while matcher/backbone stay mostly frozen.",
        3: "Stage 3: train dustbin only (matcher/backbone/K frozen).",
        4: "Stage 4: joint training with matcher/backbone/K/dustbin active.",
    }
    print(stage_messages.get(stage, f"Stage {stage}: fallback full fine-tuning."))

    trainable_count = sum(int(p.requires_grad) for p in model.parameters())
    total_count = sum(1 for _ in model.parameters())
    print(f"Trainable params: {trainable_count}/{total_count}")

    # =====================================================
    # Schedulers for Both Optimizers
    # =====================================================
    # milestones = [int(0.6 * num_epochs), int(0.7 * num_epochs), int(0.9 * num_epochs)]
    # milestones = [int(0.025*num_epochs),int(0.05*num_epochs),int(0.2*num_epochs),int(0.4*num_epochs),int(0.6*num_epochs),int(0.8*num_epochs),int(0.1*num_epochs)]
    warmup_epochs = WARMUP_EPOCHS
    warmup_k_epochs = WARMUP_K_EPOCHS
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            #    milestones=milestones,
                                            #    gamma=LR_DECAY,
                                            #    last_epoch=-1)    
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=LR_DECAY)
    # if stage in ( 2, 3):
    #     main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LR * 1e-3)
    # scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)
    # main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=LR * 1e-3)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)

    if optimizer_k is not None:
        main_scheduler_k = optim.lr_scheduler.ReduceLROnPlateau(optimizer_k, patience=1, factor=LR_DECAY)
        scheduler_k = WarmupScheduler(optimizer_k, warmup_epochs=warmup_k_epochs, after_scheduler=main_scheduler_k)

    # =====================================================
    # Checkpoint Loading (if start_epoch > 0)
    # =====================================================
    model_path = ""
    optim_path = ""
    optim_k_path = ""
    if start_epoch != 0:
        model_path = str(checkpoint_path / f'params_{start_epoch:04}.pt')
        optim_path = str(checkpoint_path / f'optim_{start_epoch:04}.pt')
        if optimizer_k is not None:
            optim_k_path = str(checkpoint_path / f'optim_k_{start_epoch:04}.pt')
        
    if len(PRETRAINED_PATH) > 0:
        print(f"Using explicitly configured pretrained path: {PRETRAINED_PATH}")
        model_path = PRETRAINED_PATH
    elif start_epoch == 0:
        default_pretrained = _default_pretrained_path(stage_output_paths, stage)
        if default_pretrained and Path(default_pretrained).exists():
            model_path = default_pretrained
        elif default_pretrained:
            print(f"[Stage {stage}] Default preload not found, starting fresh: {default_pretrained}")

    if len(model_path) > 0:
        print("Loading model parameters from {}".format(model_path))
        load_model(model, model_path)
    if len(optim_path) > 0:
        print("Loading optimizer state from {}".format(optim_path))
        load_optimizer(optimizer, optim_path)

    if len(optim_k_path) > 0:
        print("Loading optimizer_k state from {}".format(optim_k_path))
        load_optimizer(optimizer_k, optim_k_path)
    # WarmupScheduler already applies the initial warmup step when it is created.
    # Avoid manually scaling LRs here, or the first-epoch LR is reduced twice.
    print("Warmup schedulers initialized; keeping scheduler-managed starting learning rates.")


    # best_model_path = str(checkpoint_path / "best_model.pt")
    # if os.path.exists(best_model_path):
    #     print(f"Loading best model weights from {best_model_path} before training loop...")
    #     load_model(model, best_model_path)
                
    
    last_epoch = 0
    # =====================================================
    # Training Loop
    # =====================================================
    for epoch in range(start_epoch, start_epoch + num_epochs):
        logger.info(f"Epoch {epoch}/{start_epoch + num_epochs - 1}")
        logger.info("-" * 50)
        print("Epoch {}/{}".format(epoch, start_epoch + num_epochs - 1))
        print("-" * 10)

        model.train()
        if stage in (2, 3):
            # Backbone is frozen in these stages — keep BN running stats fixed.
            model.apply(_set_batchnorm_eval)
        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print("K_regression_lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer_k.param_groups]))

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)

        if optimizer_k is not None:
            for i, param_group in enumerate(optimizer_k.param_groups):
                writer.add_scalar(f'Learning_Rate_K/group_{i}', param_group['lr'], epoch)


        # Train for one epoch
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
            detect_anomaly=DETECT_ANOMALY,
            max_iters=num_iterations,
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
            stage,
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
        if stage == 2:
            scheduler.step(avg_ks_loss)
            # scheduler.step()
        elif stage == 3:
            scheduler.step(avg_val_total)
            # scheduler.step()

        else:
            scheduler.step(avg_val_loss)
            # scheduler.step()

        if optimizer_k is not None:
            # scheduler_k.step()
            scheduler_k.step(avg_ks_loss)   


        # Detect LR reduction for main optimizer
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        lr_reduced = any(clr < plr for clr, plr in zip(curr_lr, prev_lr))
        prev_lr = curr_lr  # Update previous for next iteration

        # if lr_reduced:
        #     print("[LR REDUCED] Reloading best model weights from", checkpoint_path / "best_model.pt")
        #     best_model_path = str(checkpoint_path / "best_model.pt")
        #     load_model(model, best_model_path)

        last_epoch = epoch
            
       
        
    # ---- Test Evaluation Periodically ----
    test_evaluation(
        model,
        test_dataloader,
        criterion,
        device,
        writer,
        last_epoch,
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

strip_dustbin_from_outputs(outputs)
    
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

n1, n2 = ns_pair_to_ints(outputs, sample_idx=0)
kp0 = kp0[:n1]
kp1 = kp1[:n2]

# Ensure keypoints lists for OpenCV are correctly formed:
cv2_kp0 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp0]
cv2_kp1 = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in kp1]

ds_mat = outputs["ds_mat"][0, :n1, :n2].cpu().numpy()
per_mat = outputs["perm_mat"][0, :n1, :n2].cpu().numpy()

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
