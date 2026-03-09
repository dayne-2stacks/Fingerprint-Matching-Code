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
from torch.nn.modules.batchnorm import _BatchNorm


from src.train.data_loader import build_dataloaders
from src.train.training_loop import train_epoch
from src.train.evaluation import validate_epoch, test_evaluation
from src.model.ngm import (
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
# from apex import amp


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

start_epoch = float('inf')


def _set_batchnorm_eval(module):
    """Keep BN running stats fixed when backbone/matcher are frozen."""
    if isinstance(module, _BatchNorm):
        module.eval()


def _stage_from_filename(name: str) -> int:
    name = name.lower()
    if "stage1" in name:
        return 1
    if "stage2" in name:
        return 2
    if "stage3" in name:
        return 3
    if "stage4" in name:
        return 4
    return 0


def _stage_group_label(stage: int) -> str:
    stage = int(stage)
    if stage == 1:
        return "shared_matcher"
    if stage == 2:
        return "topk"
    if stage == 3:
        return "dustbin"
    if stage == 4:
        return "joint"
    return "full"


# STAGE_CONFIG_FILES = ["stage1.yml", "stage2.yml", "stage3.yml", "stage4.yml"]
# STAGE_CONFIG_FILES = ["stage1.yml"]
# STAGE_CONFIG_FILES = ["stage2.yml", "stage3.yml"]
STAGE_CONFIG_FILES = ["stage4.yml"]

GLOBAL_CONFIG_FILE = "config.yml"


def _load_global_config(cfg_file=GLOBAL_CONFIG_FILE):
    with open(cfg_file, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    return {
        "train_defaults": raw_cfg.get("train_defaults", {}),
        "policy_defaults": raw_cfg.get("policy_defaults", {}),
        "data_defaults": raw_cfg.get("data_defaults", {}),
    }


def _load_stage_config(cfg_file, global_cfg):
    with open(cfg_file, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    train_cfg = raw_cfg.get("train", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    ngm_cfg = raw_cfg.get("ngm", {})
    if not isinstance(ngm_cfg, dict):
        ngm_cfg = {}

    cfg = dict(global_cfg.get("train_defaults", {}))
    cfg.update(global_cfg.get("policy_defaults", {}))
    cfg.update(train_cfg)
    cfg.update(
        {
            "REGRESSION": bool(ngm_cfg.get("REGRESSION", False)),
            "DUSTBIN_REJECT_ENABLE": bool(ngm_cfg.get("DUSTBIN_REJECT_ENABLE", False)),
        }
    )
    return cfg


def _collect_param_groups(model):
    groups = {
        "matcher": [],
        "backbone_node": [],
        "backbone_edge": [],
        "k_head": [],
        "dustbin": [],
    }

    for name, param in model.named_parameters():
        if name.startswith(("encoder_k.", "final_row.", "final_col.")):
            groups["k_head"].append(param)
        elif name == "bin_score":
            groups["dustbin"].append(param)
        elif name.startswith("node_layers."):
            groups["backbone_node"].append(param)
        elif name.startswith(("edge_layers.", "final_layers.")):
            groups["backbone_edge"].append(param)
        else:
            groups["matcher"].append(param)
    return groups


def _set_trainable(params, enabled):
    for p in params:
        p.requires_grad = bool(enabled)


def _configure_stage_trainability(model, stage):
    groups = _collect_param_groups(model)

    for _, param in model.named_parameters():
        param.requires_grad = False

    if stage == 1:
        # Stage 1: train shared matcher baseline (matcher + backbone), keep k/dustbin frozen.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["backbone_node"], True)
        _set_trainable(groups["backbone_edge"], True)

        _set_trainable(groups["k_head"], False)
        _set_trainable(groups["dustbin"], False)
    elif stage == 2:
        _set_trainable(groups["matcher"], False)
        _set_trainable(groups["backbone_node"], False)
        _set_trainable(groups["backbone_edge"], False)
        # Stage 2: keep matcher stable; train K.
        _set_trainable(groups["k_head"], True)
        _set_trainable(groups["dustbin"], False)

    elif stage == 3:
        _set_trainable(groups["matcher"], False)
        _set_trainable(groups["k_head"], False)
        _set_trainable(groups["backbone_node"], False)
        _set_trainable(groups["backbone_edge"], False)
        # Stage 3: train dustbin only.
        _set_trainable(groups["dustbin"], True)
    elif stage == 4:
        # Stage 4: joint fine-tuning of all modules.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["k_head"], True)
        _set_trainable(groups["backbone_node"], True)
        _set_trainable(groups["backbone_edge"], True)
        _set_trainable(groups["dustbin"], True)
    else:
        # Fallback: full fine-tuning.
        for _, param in model.named_parameters():
            param.requires_grad = True

    return groups


def _build_optimizers(model, groups, lr, backbone_lr, k_lr):
    k_params = [p for p in groups["k_head"] if p.requires_grad]
    k_ids = {id(p) for p in k_params}

    backbone_params = [
        p
        for p in (groups["backbone_node"] + groups["backbone_edge"])
        if p.requires_grad and id(p) not in k_ids
    ]
    backbone_ids = {id(p) for p in backbone_params}

    main_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in k_ids and id(p) not in backbone_ids
    ]

    model_param_groups = []
    if main_params:
        model_param_groups.append(
            {"params": main_params, "lr": float(lr), "base_lr": float(lr), "name": "main"}
        )
    if backbone_params:
        model_param_groups.append(
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "base_lr": float(backbone_lr),
                "name": "backbone",
            }
        )

    if not model_param_groups:
        if k_params:
            # Some stages (e.g. stage2 after removing the auth head) only train the K head.
            # In that case, fall back to a single optimizer over the K params to keep the
            # training loop contract unchanged.
            optimizer = optim.AdamW(
                [{"params": k_params, "lr": float(k_lr), "base_lr": float(k_lr), "name": "k_head"}],
                lr=float(k_lr),
                weight_decay=1e-4,
            )
            return optimizer, None
        raise RuntimeError("No trainable parameters were selected for the main optimizer.")

    optimizer = optim.AdamW(model_param_groups, lr=float(lr), weight_decay=1e-4)
    optimizer_k = None
    if k_params:
        optimizer_k = optim.AdamW(
            [{"params": k_params, "lr": float(k_lr), "base_lr": float(k_lr), "name": "k_head"}],
            lr=float(k_lr),
            weight_decay=1e-4,
        )
    return optimizer, optimizer_k


def _default_pretrained_path(stage_output_paths, stage):
    chain = {
        2: 1,
        3: 2,
        4: 3,
    }
    source_stage = chain.get(int(stage))
    if source_stage is None:
        return ""
    source_output_path = stage_output_paths.get(int(source_stage), "")
    if not source_output_path:
        return ""
    print(f"Stage {stage} default pretrained path: {source_output_path}")
    return str(Path(source_output_path) / "params" / "best_model.pt")


def _collect_stage_output_paths(config_file_list, stage_configs):
    stage_output_paths = {}
    for cfg_file in config_file_list:
        stage = _stage_from_filename(cfg_file)
        if stage <= 0:
            continue
        stage_output_paths[int(stage)] = stage_configs[cfg_file]["OUTPUT_PATH"]
    return stage_output_paths


global_cfg = _load_global_config(GLOBAL_CONFIG_FILE)
config_files = list(STAGE_CONFIG_FILES)
stage_configs = {cfg_file: _load_stage_config(cfg_file, global_cfg) for cfg_file in config_files}
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
    TRAIN_USE_PRED_K = bool(stage_cfg["TRAIN_USE_PRED_K"])
    DUSTBIN_LOSS_WEIGHT = float(stage_cfg["DUSTBIN_LOSS_WEIGHT"])
    DUSTBIN_K_MSE_WEIGHT = float(stage_cfg.get("DUSTBIN_K_MSE_WEIGHT", 1.0))
    DUSTBIN_REJECT_ENABLE = bool(stage_cfg["DUSTBIN_REJECT_ENABLE"])
    DUSTBIN_REJECT_MARGIN = float(stage_cfg["DUSTBIN_REJECT_MARGIN"])
    DETECT_ANOMALY = bool(stage_cfg["DETECT_ANOMALY"])
    FOCAL_GAMMA = float(stage_cfg.get("FOCAL_GAMMA", 1.0))
    STAGE1_SS_BASE_AUX_WEIGHT = float(stage_cfg.get("STAGE1_SS_BASE_AUX_WEIGHT", 0.25))

    print("BACKBONE_LR =", BACKBONE_LR)
    print("Start epoch: ", start_epoch)
    if stage == 1:
        print("STAGE1_SS_BASE_AUX_WEIGHT =", STAGE1_SS_BASE_AUX_WEIGHT)

    
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
    if DUSTBIN_LOSS_WEIGHT >= DUSTBIN_K_MSE_WEIGHT:
        warn_msg = (
            "Configured dustbin BCE weight is not lower than OT-K MSE weight: "
            f"DUSTBIN_LOSS_WEIGHT={DUSTBIN_LOSS_WEIGHT}, "
            f"DUSTBIN_K_MSE_WEIGHT={DUSTBIN_K_MSE_WEIGHT}. "
            "This is allowed, but the intended default is BCE < OT-K MSE."
        )
        print(f"[WARN] {warn_msg}")
        logger.warning(warn_msg)


   

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
    )
    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(
        regression=REGRESSION,
        dustbin_loss_weight=DUSTBIN_LOSS_WEIGHT,
        dustbin_k_mse_weight=DUSTBIN_K_MSE_WEIGHT,
        dustbin_reject_enable=DUSTBIN_REJECT_ENABLE,
        dustbin_reject_margin=DUSTBIN_REJECT_MARGIN,
        train_use_pred_k=TRAIN_USE_PRED_K,
    )

    # criterion = FocalLoss(gamma=FOCAL_GAMMA)
    criterion = PermutationLossHung()
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
    # main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=LR_DECAY)
    # if stage in ( 2, 3):
    #     main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LR * 1e-3)
    # scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=LR * 1e-3)
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
    # Initialize warmup scheduler learning rates after loading optimizer state
    # if start_epoch == 0:  # Only for fresh training, not resuming
    print("Initializing warmup learning rates for first epoch...")
    warmup_main = max(int(warmup_epochs), 1)
    for param_group in optimizer.param_groups:
        base_lr = float(param_group.get("base_lr", param_group.get("lr", LR)))
        param_group['lr'] = base_lr / warmup_main
    
    if optimizer_k is not None and scheduler_k is not None:
        warmup_k = max(int(warmup_k_epochs), 1)
        for param_group in optimizer_k.param_groups:
            base_lr = float(param_group.get("base_lr", param_group.get("lr", K_LR)))
            param_group['lr'] = base_lr / warmup_k


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
        # if stage in (1, 2, 3):
        #     # Stages with partial freezing should keep BN running stats fixed.
        #     model.apply(_set_batchnorm_eval)
        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print("K_regression_lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer_k.param_groups]))

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)

        if optimizer_k is not None:
            for i, param_group in enumerate(optimizer_k.param_groups):
                writer.add_scalar(f'Learning_Rate_K/group_{i}', param_group['lr'], epoch)

        writer.add_scalar('Train/Train_Use_Pred_K', float(bool(TRAIN_USE_PRED_K)), epoch)

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
            stage1_ss_base_aux_weight=STAGE1_SS_BASE_AUX_WEIGHT,
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
            stage1_ss_base_aux_weight=STAGE1_SS_BASE_AUX_WEIGHT,
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
            # scheduler.step(avg_ks_loss)
            scheduler.step()
        elif stage == 3:
            # scheduler.step(avg_val_total)
            scheduler.step()

        else:
            # scheduler.step(avg_val_loss)
            scheduler.step()

        if optimizer_k is not None:
            # scheduler_k.step()
            scheduler_k.step(avg_ks_loss)   


        # Detect LR reduction for main optimizer
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        lr_reduced = any(clr < plr for clr, plr in zip(curr_lr, prev_lr))
        prev_lr = curr_lr  # Update previous for next iteration

        if lr_reduced:
            print("[LR REDUCED] Reloading best model weights from", checkpoint_path / "best_model.pt")
            best_model_path = str(checkpoint_path / "best_model.pt")
            load_model(model, best_model_path)

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
