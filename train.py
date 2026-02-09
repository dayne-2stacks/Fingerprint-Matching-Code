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
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from src.parallel import DataParallel
from src.loss_func import PermutationLoss, PermutationLossHung, FocalLoss
from utils.models_sl import save_model, load_model, load_optimizer
from utils.visualize import visualize_stochastic_matrix, visualize_match, to_grayscale_cv2_image
from src.evaluation_metric import matching_accuracy
from utils.scheduler import WarmupScheduler
from src.model.dustbin import strip_dustbin_by_ns
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


STAGE_DEFAULTS = {
    1: {
        "LR": 1e-4,
        "K_LR": 2e-3,
        "BACKBONE_LR": 1e-5,
        "LR_DECAY": 0.5,
        "patience": 15,
        "num_epochs": 50,
        "num_iterations": 50,
        "BATCH_SIZE": 8,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 0.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": False,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 0.0,
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": False,
        "AUTH_GATE_THRESH": 0.5,
        "WARMUP_EPOCHS": 5,
        "WARMUP_K_EPOCHS": 3,
        "DETECT_ANOMALY": False,
    },
    2: {
        "LR": 1e-20,
        "K_LR": 1e-6,
        "BACKBONE_LR": 1e-20,
        "LR_DECAY": 0.1,
        "patience": 10,
        "num_epochs": 1,
        "num_iterations": 25,
        "BATCH_SIZE": 8,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 0.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": False,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 0.0,
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": False,
        "AUTH_GATE_THRESH": 0.5,
        "WARMUP_EPOCHS": 5,
        "WARMUP_K_EPOCHS": 3,
        "DETECT_ANOMALY": False,
    },
    3: {
        "LR": 1e-4,
        "K_LR": 1e-4,
        "BACKBONE_LR": 1e-20,
        "LR_DECAY": 0.5,
        "patience": 5,
        "num_epochs": 20,
        "num_iterations": 25,
        "BATCH_SIZE": 8,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 1.0,
        "SG_DUSTBIN_RAMP": {"start": 0.1, "end": 1.0},
        "DUSTBIN_REJECT_ENABLE": True,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 0.0,
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": False,
        "AUTH_GATE_THRESH": 0.5,
        "WARMUP_EPOCHS": 5,
        "WARMUP_K_EPOCHS": 3,
        "DETECT_ANOMALY": False,
    },
    4: {
        "LR": 1e-5,
        "K_LR": 1e-4,
        "BACKBONE_LR": 1e-6,
        "LR_DECAY": 0.5,
        "patience": 5,
        "num_epochs": 20,
        "num_iterations": 25,
        "BATCH_SIZE": 8,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 1.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": True,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 1.0,
        "K_GATE_ENABLE": True,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": True,
        "AUTH_GATE_THRESH": 0.5,
        "WARMUP_EPOCHS": 5,
        "WARMUP_K_EPOCHS": 3,
        "DETECT_ANOMALY": False,
    },
}
config_files = ["stage2.yml", "stage3.yml", "stage4.yml"]
# config_files = ["stage1.yml", "stage2.yml", "stage3.yml"]
# config_files = ["stage4.yml"]
# config_files = [ "stage2.yml"]
# config_files = [ "stage4.yml"]


for file in config_files:
    scheduler = scheduler_k = None
    print("Using config ", file)

    # ====================================================
    # Load Settings from YAML Configuration File
    # =====================================================
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]
    stage = _stage_from_filename(file)
    defaults = STAGE_DEFAULTS.get(stage, STAGE_DEFAULTS[1])

    OUTPUT_PATH = train_config.get("OUTPUT_PATH", train_config.get("MODEL_PATH", "results/binary-classifier"))
    PRETRAINED_PATH = train_config.get("PRETRAINED_PATH", "")
    CHECKPOINT_PATH = train_config.get("CHECKPOINT_PATH", "checkpoints")
    checkpoint_root = Path(CHECKPOINT_PATH)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    start_file = checkpoint_root / "checkpoint.json"
    log_dir = Path(train_config.get("LOG_DIR", "logs/tensorboard"))
    log_dir.mkdir(parents=True, exist_ok=True)

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

    num_iterations = train_config.get("num_iterations", defaults["num_iterations"])
    num_epochs = train_config.get("num_epochs", defaults["num_epochs"])
    BATCH_SIZE = train_config.get("BATCH_SIZE", defaults["BATCH_SIZE"])
    BM_NAME = train_config.get("BM_NAME", "L3SFV2AugmentedBenchmark")
    FILTER = train_config.get("FILTER", None)
    LR = train_config.get("LR", defaults["LR"])
    BACKBONE_LR = defaults["BACKBONE_LR"]
    K_LR = train_config.get("K_LR", defaults["K_LR"])
    SG_DUSTBIN_WEIGHT = defaults["SG_DUSTBIN_WEIGHT"]
    SG_DUSTBIN_RAMP = defaults["SG_DUSTBIN_RAMP"]
    DUSTBIN_REJECT_ENABLE = defaults["DUSTBIN_REJECT_ENABLE"]
    DUSTBIN_REJECT_MARGIN = defaults["DUSTBIN_REJECT_MARGIN"]
    LR_DECAY = defaults["LR_DECAY"]
    patience = defaults["patience"]
    ngm_config = config.get("ngm", {})
    REGRESSION = ngm_config.get("REGRESSION", stage != 1)
    TRAIN_USE_PRED_K = ngm_config.get("TRAIN_USE_PRED_K", stage == 4)
    PERM_LOSS = defaults["PERM_LOSS"]
    FOCAL_GAMMA = defaults["FOCAL_GAMMA"]
    K_REG_WEIGHT = defaults["K_REG_WEIGHT"]
    K_CLS_WEIGHT = defaults["K_CLS_WEIGHT"]
    DUSTBIN_LOSS_WEIGHT = defaults["DUSTBIN_LOSS_WEIGHT"]
    AUTH_WEIGHT = defaults["AUTH_WEIGHT"]
    K_GATE_ENABLE = defaults["K_GATE_ENABLE"]
    K_GATE_THRESH = defaults["K_GATE_THRESH"]
    K_MATCH_ROUNDING = defaults["K_MATCH_ROUNDING"]
    AUTH_GATE_ENABLE = defaults["AUTH_GATE_ENABLE"]
    AUTH_GATE_THRESH = defaults["AUTH_GATE_THRESH"]
    DETECT_ANOMALY = defaults["DETECT_ANOMALY"]

    print("BACKBONE_LR =", BACKBONE_LR)
    print("Start epoch: ", start_epoch)
    minimal_cfg = {
        "LR": LR,
        "K_LR": K_LR,
        "num_epochs": num_epochs,
        "num_iterations": num_iterations,
        "BATCH_SIZE": BATCH_SIZE,
        "BM_NAME": BM_NAME,
        "OUTPUT_PATH": OUTPUT_PATH,
        "PRETRAINED_PATH": PRETRAINED_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "REGRESSION": REGRESSION,
        "TRAIN_USE_PRED_K": TRAIN_USE_PRED_K,
    }
    derived_cfg = {
        "BACKBONE_LR": BACKBONE_LR,
        "LR_DECAY": LR_DECAY,
        "patience": patience,
        "PERM_LOSS": PERM_LOSS,
        "FOCAL_GAMMA": FOCAL_GAMMA,
        "K_REG_WEIGHT": K_REG_WEIGHT,
        "K_CLS_WEIGHT": K_CLS_WEIGHT,
        "DUSTBIN_LOSS_WEIGHT": DUSTBIN_LOSS_WEIGHT,
        "SG_DUSTBIN_WEIGHT": SG_DUSTBIN_WEIGHT,
        "SG_DUSTBIN_RAMP": SG_DUSTBIN_RAMP,
        "DUSTBIN_REJECT_ENABLE": DUSTBIN_REJECT_ENABLE,
        "DUSTBIN_REJECT_MARGIN": DUSTBIN_REJECT_MARGIN,
        "AUTH_WEIGHT": AUTH_WEIGHT,
        "K_GATE_ENABLE": K_GATE_ENABLE,
        "K_GATE_THRESH": K_GATE_THRESH,
        "K_MATCH_ROUNDING": K_MATCH_ROUNDING,
        "AUTH_GATE_ENABLE": AUTH_GATE_ENABLE,
        "AUTH_GATE_THRESH": AUTH_GATE_THRESH,
        "WARMUP_EPOCHS": defaults["WARMUP_EPOCHS"],
        "WARMUP_K_EPOCHS": defaults["WARMUP_K_EPOCHS"],
        "DETECT_ANOMALY": DETECT_ANOMALY,
    }
    print(f"[Stage {stage}] config={minimal_cfg} defaults={derived_cfg}")
    
    # =====================================================
    # Hard-Coded and Derived Parameters
    # =====================================================
    dataset_len = 640

    best_loss = float('inf')
    no_improvement_count = 0

    # File paths
    # Default to the synthetic dataset; override for stage 6
    train_root = 'dataset/Synthetic'
    # OUTPUT_PATH = "results/base"

    # =====================================================
    # Setup Logging
    # =====================================================
    logging.basicConfig(
        filename='fp.log', 
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    logger.info("[Stage %s] config=%s defaults=%s", stage, minimal_cfg, derived_cfg)

   

    # =====================================================
    # Dataset and Dataloader
    # =====================================================
    # task = 'classify' if stage in (4, 5, 6) else 'match'

    # if stage == 6:
    #     # Use L3SF session/identity-based pairing for stage 6
    #     train_root = 'dataset/L3-SF'
    #     dataset_kind = 'l3sf'
    dataloader, val_dataloader, test_dataloader = build_dataloaders(
        train_root,
        dataset_len,
        batch_size=BATCH_SIZE,
        benchmark_name=BM_NAME,
        filter=FILTER,
    )
    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(
        regression=REGRESSION,
        k_reg_weight=K_REG_WEIGHT,
        k_cls_weight=K_CLS_WEIGHT,
        dustbin_loss_weight=DUSTBIN_LOSS_WEIGHT,
        k_gate_enable=K_GATE_ENABLE,
        k_gate_thresh=K_GATE_THRESH,
        k_match_rounding=K_MATCH_ROUNDING,
        auth_gate_enable=AUTH_GATE_ENABLE,
        auth_gate_thresh=AUTH_GATE_THRESH,
    )
    model.dustbin_reject_enable = bool(DUSTBIN_REJECT_ENABLE)
    model.dustbin_reject_margin = float(DUSTBIN_REJECT_MARGIN)
    model.train_use_pred_k = bool(TRAIN_USE_PRED_K)
    model.train_stage = int(stage) if stage is not None else None
    perm_loss_key = str(PERM_LOSS).strip().lower()
    if perm_loss_key == "focal":
        criterion = FocalLoss(gamma=float(FOCAL_GAMMA))
    elif perm_loss_key in {"hung", "hungarian"}:
        criterion = PermutationLossHung()
    else:
        criterion = PermutationLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Uncomment below if using multiple GPUs:
    # model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # model, optimizer = amp.initialize(model, optimizer)


    # =====================================================
    # Freeze / Unfreeze Layers Based on Stage
    # =====================================================

    backbone_ids = [id(item) for item in model.backbone_params]
    k_params = model.k_params_id
    dustbin_params = [model.bin_score]
    dustbin_ids = {id(param) for param in dustbin_params}
    other_params = [
        param for param in model.parameters()
        if id(param) not in k_params
        and id(param) not in backbone_ids
        and id(param) not in dustbin_ids
    ]
    
    model_params = [
        {'params': other_params},
        {'params': model.backbone_params, 'lr': BACKBONE_LR},
        {'params': dustbin_params},
        ]
    
    # -----------------------------------------------------
    # Determine training stage from config filename
    # -----------------------------------------------------
    if stage == 1:
        print("Stage 1: Freezing all layers in k_params and training other parameters.")
        # Freeze k_params
        # for param in model.k_params:
        #     for p in param["params"]:
        #         p.requires_grad = False

        for name, param in model.named_parameters():
            if id(param) in model.k_params_id:
                param.requires_grad = False
            else:
                param.requires_grad = True

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

                # In stage 2, only k_params are trainable.
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)

    elif stage == 3:
        print("Stage 3: Train dustbin and k_params (matcher/backbone frozen).")
        for name, param in model.named_parameters():
            if id(param) in model.k_params_id or id(param) in dustbin_ids:
                param.requires_grad = True
            else:
                param.requires_grad = False

                # In stage 3, train dustbin and k_params.
        optimizer = optim.AdamW(dustbin_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)


    # elif "stage3" in file:
    #     stage = 3
    #     # Stage 3: All parameters are trainable. We separate backbone parameters for a different LR.s
    #     print("Stage 3: Unfreezing all layers for full fine-tuning.")
    #     # Unfreeze every parameter
    #     for name,  param in model.named_parameters():
    #         param.requires_grad = True


    #     optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
    #     optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-4)
    elif stage == 4:
        print("Stage 4: Joint fine-tuning (all parameters trainable).")
        for _, param in model.named_parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
        optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)
    else:
        stage = None



      

    # elif stage == 4:
    #     print("Stage 4: Classification training, optimizing only k parameters.")
    #     for name, param in model.named_parameters():
    #         if id(param) not in model.k_params_id:
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True
    #     optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
    #     optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)
    # elif stage == 5:
    #     print("Stage 5: Classification training, optimizing K, graph matcher, and backbone (freeze classifier).")
    #     # Train everything except the match classifier to learn from negatives
    #     optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
    #     optimizer_k = optim.AdamW(model.k_params, lr=K_LR, weight_decay=1e-6)
    # elif stage == 6:
    #     print("Stage 6: Training match classifier only.")
       
    #     optimizer = optim.AdamW(model_params, lr=LR, weight_decay=1e-4)
    #     optimizer_k = None
    # else:
    #     optimizer = optim.AdamW(model.parameters(), lr=LR)
    #     optimizer_k = None

    # =====================================================
    # Schedulers for Both Optimizers
    # =====================================================
    # milestones = [int(0.6 * num_epochs), int(0.7 * num_epochs), int(0.9 * num_epochs)]
    # milestones = [int(0.025*num_epochs),int(0.05*num_epochs),int(0.2*num_epochs),int(0.4*num_epochs),int(0.6*num_epochs),int(0.8*num_epochs),int(0.1*num_epochs)]
    warmup_epochs = defaults["WARMUP_EPOCHS"]
    warmup_k_epochs = defaults["WARMUP_K_EPOCHS"]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            #    milestones=milestones,
                                            #    gamma=LR_DECAY,
                                            #    last_epoch=-1)
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=LR_DECAY)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)
    if optimizer_k is not None:
        main_scheduler_k = optim.lr_scheduler.ReduceLROnPlateau(optimizer_k, patience=1, factor=LR_DECAY)
        scheduler_k = WarmupScheduler(optimizer_k, warmup_epochs=warmup_k_epochs, after_scheduler=main_scheduler_k)

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
        load_optimizer(optimizer, optim_path)

    if len(optim_k_path) > 0:
        print("Loading optimizer_k state from {}".format(optim_k_path))
        load_optimizer(optimizer_k, optim_k_path)
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
        initial_k_lr = scheduler_k.get_initial_lr() if hasattr(scheduler_k, 'get_initial_lr') else K_LR / warmup_k_epochs
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
        if stage in (2, 3):
            # Stage 2/3 freeze most of the network; avoid BN train/eval drift.
            model.apply(_set_batchnorm_eval)
        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print("K_regression_lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer_k.param_groups]))

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)

        if optimizer_k is not None:
            for i, param_group in enumerate(optimizer_k.param_groups):
                writer.add_scalar(f'Learning_Rate_K/group_{i}', param_group['lr'], epoch)

        # Compute SG dustbin weight for this epoch (optional ramp)
        sg_weight = SG_DUSTBIN_WEIGHT
        if isinstance(SG_DUSTBIN_RAMP, dict):
            ramp_start = float(SG_DUSTBIN_RAMP.get("start", sg_weight))
            ramp_end = float(SG_DUSTBIN_RAMP.get("end", sg_weight))
            denom = max(num_epochs - 1, 1)
            progress = (epoch - start_epoch) / denom
            sg_weight = ramp_start + (ramp_end - ramp_start) * max(0.0, min(1.0, progress))

        writer.add_scalar('Train/SG_Dustbin_Weight', sg_weight, epoch)
        k_pred_ramp = 0.0
        if stage == 4 and TRAIN_USE_PRED_K:
            denom = max(num_epochs - 1, 1)
            k_pred_ramp = (epoch - start_epoch) / denom
            k_pred_ramp = max(0.0, min(1.0, k_pred_ramp))
        model.k_pred_ramp = float(k_pred_ramp)
        writer.add_scalar('Train/K_Pred_Ramp', model.k_pred_ramp, epoch)

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
            sg_dustbin_weight=sg_weight,
            auth_weight=AUTH_WEIGHT,
            stage2_full_loss=bool(TRAIN_USE_PRED_K),
            detect_anomaly=DETECT_ANOMALY,
            max_iters=num_iterations,
        )
            
        # =====================================================
        # ---- Validation after each epoch ----
        # =====================================================
        avg_val_loss, avg_ks_loss, avg_val_total, avg_val_accuracy, auth_threshold = validate_epoch(
            model,
            val_dataloader,
            criterion,
            device,
            writer,
            epoch,
            logger,
            stage,
            sg_dustbin_weight=sg_weight,
            auth_weight=AUTH_WEIGHT,
            stage2_full_loss=bool(TRAIN_USE_PRED_K),
        )
    
        
        # Save best model based on validation loss and update checkpoint file
        if avg_val_total < best_loss:
            best_loss = avg_val_total
            no_improvement_count = 0
            best_model_path = str(checkpoint_path / "best_model.pt")
            save_model(model, best_model_path)
            with open(start_file, "w") as f:
                json.dump({"start_epoch": epoch + 1}, f)
            if stage == 4:
                thresh_path = checkpoint_path / "auth_threshold.json"
                with open(thresh_path, "w") as f:
                    json.dump({"auth_threshold": float(auth_threshold)}, f)
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
        if epoch % 10 == 9:
            test_evaluation(
                model,
                test_dataloader,
                criterion,
                device,
                writer,
                epoch,
                stage,
                auth_threshold=auth_threshold,
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

if outputs.get("has_dustbin", False):
    n1 = outputs["ns"][0] - 1
    n2 = outputs["ns"][1] - 1
else:
    n1 = outputs["ns"][0]
    n2 = outputs["ns"][1]

outputs["ds_mat"] = strip_dustbin_by_ns(outputs["ds_mat"], n1, n2)
outputs["perm_mat"] = strip_dustbin_by_ns(outputs["perm_mat"], n1, n2)
if "gt_perm_mat" in outputs:
    outputs["gt_perm_mat"] = strip_dustbin_by_ns(outputs["gt_perm_mat"], n1, n2)
outputs["ns"] = [n1, n2]
    
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

n1 = outputs["ns"][0]
n2 = outputs["ns"][1]
if isinstance(n1, torch.Tensor):
    n1 = int(n1[0].item()) if n1.dim() > 0 else int(n1.item())
else:
    n1 = int(n1)
if isinstance(n2, torch.Tensor):
    n2 = int(n2[0].item()) if n2.dim() > 0 else int(n2.item())
else:
    n2 = int(n2)
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
