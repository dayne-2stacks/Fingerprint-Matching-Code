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
    K_GATE_TEMP,
    K_GATE_LOSS_WEIGHT,
    K_GATE_LEARN_STAGE_MIN,
    AUTH_GATE_TEMP,
    AUTH_GATE_LOSS_WEIGHT,
    DUSTBIN_MARGIN_TEMP,
    DUSTBIN_MARGIN_LOSS_WEIGHT,
)
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
    if "stage5" in name:
        return 5
    return 0


def _ramp_value(ramp_cfg, progress, default_value):
    if not isinstance(ramp_cfg, dict):
        return float(default_value)
    start = float(ramp_cfg.get("start", default_value))
    end = float(ramp_cfg.get("end", default_value))
    p = max(0.0, min(1.0, float(progress)))
    return start + (end - start) * p


def _collect_param_groups(model):
    groups = {
        "matcher": [],
        "backbone_node": [],
        "backbone_edge": [],
        "k_head": [],
        "auth_head": [],
        "dustbin": [],
    }

    for name, param in model.named_parameters():
        if name.startswith(("encoder_k.", "final_row.", "final_col.")):
            groups["k_head"].append(param)
        elif name.startswith(("auth_head.", "auth_pool_proj.")):
            groups["auth_head"].append(param)
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
        # Stage 1: train shared matcher baseline (matcher + backbone), keep k/auth/dustbin frozen.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["backbone_node"], True)
        _set_trainable(groups["backbone_edge"], True)
    elif stage == 2:
        # Stage 2: keep matcher stable; train K + auth heads.
        _set_trainable(groups["k_head"], True)
        _set_trainable(groups["auth_head"], True)
    elif stage == 3:
        # Stage 3: joint training for matcher/backbone/k/auth.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["k_head"], True)
        _set_trainable(groups["auth_head"], True)
        _set_trainable(groups["backbone_node"], True)
        _set_trainable(groups["backbone_edge"], True)
    elif stage == 4:
        # Stage 4: dustbin + auth warmup using shared matcher from stage1.
        _set_trainable(groups["dustbin"], True)
        _set_trainable(groups["auth_head"], True)
    elif stage == 5:
        # Stage 5: joint matcher/backbone/dustbin/auth while K stays frozen.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["auth_head"], True)
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
        raise RuntimeError("No trainable parameters were selected for the main optimizer.")

    optimizer = optim.AdamW(model_param_groups, lr=float(lr), weight_decay=1e-4)
    optimizer_k = None
    if k_params:
        optimizer_k = optim.AdamW(
            [{"params": k_params, "lr": float(k_lr), "base_lr": float(k_lr), "name": "k_head"}],
            lr=float(k_lr),
            weight_decay=1e-6,
        )

    return optimizer, optimizer_k


STAGE_RUNTIME_DEFAULTS = {
    1: {
        "LR": 1e-4,
        "K_LR": 5e-4,
        "BACKBONE_LR": 1e-5,
        "LR_DECAY": 0.5,
        "patience": 12,
        "num_epochs": 40,
        "num_iterations": 50,
        "BATCH_SIZE": 8,
        "WARMUP_EPOCHS": 3,
        "WARMUP_K_EPOCHS": 2,
    },
    2: {
        "LR": 8e-5,
        "K_LR": 3e-4,
        "BACKBONE_LR": 2e-5,
        "LR_DECAY": 0.5,
        "patience": 20,
        "num_epochs": 30,
        "num_iterations": 50,
        "BATCH_SIZE": 8,
        "WARMUP_EPOCHS": 3,
        "WARMUP_K_EPOCHS": 2,
    },
    3: {
        "LR": 5e-5,
        "K_LR": 2e-4,
        "BACKBONE_LR": 1e-5,
        "LR_DECAY": 0.5,
        "patience": 8,
        "num_epochs": 30,
        "num_iterations": 40,
        "BATCH_SIZE": 8,
        "WARMUP_EPOCHS": 3,
        "WARMUP_K_EPOCHS": 2,
    },
    4: {
        "LR": 2e-5,
        "K_LR": 1e-4,
        "BACKBONE_LR": 5e-6,
        "LR_DECAY": 0.5,
        "patience": 8,
        "num_epochs": 25,
        "num_iterations": 40,
        "BATCH_SIZE": 8,
        "WARMUP_EPOCHS": 3,
        "WARMUP_K_EPOCHS": 2,
    },
    5: {
        "LR": 2e-5,
        "K_LR": 1e-4,
        "BACKBONE_LR": 5e-6,
        "LR_DECAY": 0.5,
        "patience": 8,
        "num_epochs": 25,
        "num_iterations": 40,
        "BATCH_SIZE": 8,
        "WARMUP_EPOCHS": 3,
        "WARMUP_K_EPOCHS": 2,
    },
}

STAGE_POLICY = {
    1: {
        "MATCH_BRANCH": "shared_matcher",
        "REGRESSION": False,
        "TRAIN_USE_PRED_K": False,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "K_CLS_RAMP": None,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 0.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": False,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 0.0,
        "AUTH_WEIGHT_RAMP": None,
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": False,
        "AUTH_GATE_THRESH": 0.5,
        "AUTH_USE_POOLED": True,
        "AUTH_SCALAR_DIM": 6,
        "K_MATCH_DETACH_STAGE_THRESHOLD": 2,
        "DETECT_ANOMALY": False,
    },
    2: {
        "MATCH_BRANCH": "topk",
        "REGRESSION": True,
        "TRAIN_USE_PRED_K": False,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.05,
        "K_CLS_RAMP": {"start": 0.0, "end": 0.05},
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 0.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": False,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 0.2,
        "AUTH_WEIGHT_RAMP": {"start": 0.05, "end": 0.2},
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": False,
        "AUTH_GATE_THRESH": 0.5,
        "AUTH_USE_POOLED": True,
        "AUTH_SCALAR_DIM": 6,
        "K_MATCH_DETACH_STAGE_THRESHOLD": 2,
        "DETECT_ANOMALY": False,
    },
    3: {
        "MATCH_BRANCH": "topk",
        "REGRESSION": True,
        "TRAIN_USE_PRED_K": True,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.1,
        "K_CLS_RAMP": {"start": 0.05, "end": 0.1},
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 1.0,
        "SG_DUSTBIN_RAMP": {"start": 0.1, "end": 1.0},
        "DUSTBIN_REJECT_ENABLE": False,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 0.5,
        "AUTH_WEIGHT_RAMP": {"start": 0.2, "end": 0.5},
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": False,
        "AUTH_GATE_THRESH": 0.5,
        "AUTH_USE_POOLED": True,
        "AUTH_SCALAR_DIM": 6,
        "K_MATCH_DETACH_STAGE_THRESHOLD": 2,
        "DETECT_ANOMALY": False,
    },
    4: {
        "MATCH_BRANCH": "dustbin",
        "REGRESSION": True,
        "TRAIN_USE_PRED_K": False,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "K_CLS_RAMP": None,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 1.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": True,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 1.0,
        "AUTH_WEIGHT_RAMP": {"start": 0.2, "end": 1.0},
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": True,
        "AUTH_GATE_THRESH": 0.5,
        "AUTH_USE_POOLED": True,
        "AUTH_SCALAR_DIM": 6,
        "K_MATCH_DETACH_STAGE_THRESHOLD": 2,
        "DETECT_ANOMALY": False,
    },
    5: {
        "MATCH_BRANCH": "dustbin",
        "REGRESSION": True,
        "TRAIN_USE_PRED_K": False,
        "PERM_LOSS": "focal",
        "FOCAL_GAMMA": 2.0,
        "K_REG_WEIGHT": 0.2,
        "K_CLS_WEIGHT": 0.0,
        "K_CLS_RAMP": None,
        "DUSTBIN_LOSS_WEIGHT": 0.5,
        "SG_DUSTBIN_WEIGHT": 1.0,
        "SG_DUSTBIN_RAMP": None,
        "DUSTBIN_REJECT_ENABLE": True,
        "DUSTBIN_REJECT_MARGIN": 0.0,
        "AUTH_WEIGHT": 1.0,
        "AUTH_WEIGHT_RAMP": {"start": 0.2, "end": 1.0},
        "K_GATE_ENABLE": False,
        "K_GATE_THRESH": 0.2,
        "K_MATCH_ROUNDING": "round",
        "AUTH_GATE_ENABLE": True,
        "AUTH_GATE_THRESH": 0.5,
        "AUTH_USE_POOLED": True,
        "AUTH_SCALAR_DIM": 6,
        "K_MATCH_DETACH_STAGE_THRESHOLD": 2,
        "DETECT_ANOMALY": False,
    },
}

TRAIN_CONFIG_KEYS_USED = {
    "OUTPUT_PATH",
    "MODEL_PATH",
    "PRETRAINED_PATH",
    "CHECKPOINT_PATH",
    "LOG_DIR",
    "start_epoch",
    "num_iterations",
    "num_epochs",
    "BATCH_SIZE",
    "BM_NAME",
    "FILTER",
    "LR",
    "BACKBONE_LR",
    "K_LR",
    "WARMUP_EPOCHS",
    "WARMUP_K_EPOCHS",
    "OVERFIT_TO_TRAIN_SPLIT",
}
NGM_CONFIG_KEYS_USED = set()


def _resolve_stage_runtime_defaults(stage):
    return dict(STAGE_RUNTIME_DEFAULTS.get(stage, STAGE_RUNTIME_DEFAULTS[1]))


def _resolve_stage_policy(stage):
    return dict(STAGE_POLICY.get(stage, STAGE_POLICY[1]))


def _default_pretrained_path(stage_output_paths, stage):
    chain = {
        2: 1,
        3: 2,
        4: 1,
        5: 4,
    }
    source_stage = chain.get(int(stage))
    if source_stage is None:
        return ""
    source_output_path = stage_output_paths.get(int(source_stage), "")
    if not source_output_path:
        return ""
    return str(Path(source_output_path) / "params" / "best_model.pt")


def _get_ignored_config_keys(train_config, ngm_config):
    ignored_train = sorted(k for k in train_config.keys() if k not in TRAIN_CONFIG_KEYS_USED)
    ignored_ngm = sorted(k for k in ngm_config.keys() if k not in NGM_CONFIG_KEYS_USED)
    return ignored_train, ignored_ngm


def _collect_stage_output_paths(config_file_list):
    stage_output_paths = {}
    for cfg_file in config_file_list:
        stage = _stage_from_filename(cfg_file)
        if stage <= 0:
            continue
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
        train_cfg = cfg.get("train", {})
        output_path = train_cfg.get("OUTPUT_PATH", train_cfg.get("MODEL_PATH", "results/binary-classifier"))
        stage_output_paths[int(stage)] = output_path
    return stage_output_paths


# config_files = ["stage1.yml", "stage2.yml", "stage3.yml", "stage4.yml", "stage5.yml"]
# config_files = ["stage1.yml", "stage2.yml", "stage3.yml"]
# config_files = ["stage4.yml"]
config_files = [ "stage2.yml"]
# config_files = [ "stage4.yml"]
stage_output_paths = _collect_stage_output_paths(config_files)


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
    runtime_defaults = _resolve_stage_runtime_defaults(stage)
    stage_policy = _resolve_stage_policy(stage)
    match_branch = str(stage_policy.get("MATCH_BRANCH", "shared_matcher")).strip().lower()
    stage_name = f"stage{stage}"
    ngm_config = config.get("ngm", {})

    OUTPUT_PATH = train_config.get("OUTPUT_PATH", train_config.get("MODEL_PATH", "results/binary-classifier"))
    stage_output_paths[int(stage)] = OUTPUT_PATH
    PRETRAINED_PATH = train_config.get("PRETRAINED_PATH", "")
    CHECKPOINT_PATH = train_config.get("CHECKPOINT_PATH", "checkpoints")
    checkpoint_root = Path(CHECKPOINT_PATH) / match_branch / stage_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    start_file = checkpoint_root / "checkpoint.json"
    log_dir = Path(train_config.get("LOG_DIR", "logs/tensorboard")) / match_branch / stage_name
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
        start_epoch = train_config.get("start_epoch", 0)

    num_iterations = train_config.get("num_iterations", runtime_defaults["num_iterations"])
    num_epochs = train_config.get("num_epochs", runtime_defaults["num_epochs"])
    BATCH_SIZE = train_config.get("BATCH_SIZE", runtime_defaults["BATCH_SIZE"])
    BM_NAME = train_config.get("BM_NAME", "L3SFV2AugmentedBenchmark")
    FILTER = train_config.get("FILTER", None)
    OVERFIT_TO_TRAIN_SPLIT = bool(train_config.get("OVERFIT_TO_TRAIN_SPLIT", False))
    LR = train_config.get("LR", runtime_defaults["LR"])
    BACKBONE_LR = train_config.get("BACKBONE_LR", runtime_defaults["BACKBONE_LR"])
    K_LR = train_config.get("K_LR", runtime_defaults["K_LR"])
    LR_DECAY = runtime_defaults["LR_DECAY"]
    patience = runtime_defaults["patience"]

    WARMUP_EPOCHS = train_config.get("WARMUP_EPOCHS", runtime_defaults["WARMUP_EPOCHS"])
    WARMUP_K_EPOCHS = train_config.get("WARMUP_K_EPOCHS", runtime_defaults["WARMUP_K_EPOCHS"])

    REGRESSION = stage_policy["REGRESSION"]
    TRAIN_USE_PRED_K = stage_policy["TRAIN_USE_PRED_K"]
    PERM_LOSS = stage_policy["PERM_LOSS"]
    FOCAL_GAMMA = stage_policy["FOCAL_GAMMA"]
    K_REG_WEIGHT = stage_policy["K_REG_WEIGHT"]
    K_CLS_WEIGHT = stage_policy["K_CLS_WEIGHT"]
    K_CLS_RAMP = stage_policy["K_CLS_RAMP"]
    DUSTBIN_LOSS_WEIGHT = stage_policy["DUSTBIN_LOSS_WEIGHT"]
    SG_DUSTBIN_WEIGHT = stage_policy["SG_DUSTBIN_WEIGHT"]
    SG_DUSTBIN_RAMP = stage_policy["SG_DUSTBIN_RAMP"]
    DUSTBIN_REJECT_ENABLE = stage_policy["DUSTBIN_REJECT_ENABLE"]
    DUSTBIN_REJECT_MARGIN = stage_policy["DUSTBIN_REJECT_MARGIN"]
    AUTH_WEIGHT = stage_policy["AUTH_WEIGHT"]
    AUTH_WEIGHT_RAMP = stage_policy["AUTH_WEIGHT_RAMP"]
    K_GATE_ENABLE = stage_policy["K_GATE_ENABLE"]
    K_GATE_THRESH = stage_policy["K_GATE_THRESH"]
    K_MATCH_ROUNDING = stage_policy["K_MATCH_ROUNDING"]
    AUTH_GATE_ENABLE = stage_policy["AUTH_GATE_ENABLE"]
    AUTH_GATE_THRESH = stage_policy["AUTH_GATE_THRESH"]
    AUTH_USE_POOLED = stage_policy["AUTH_USE_POOLED"]
    AUTH_SCALAR_DIM = stage_policy["AUTH_SCALAR_DIM"]
    K_MATCH_DETACH_STAGE_THRESHOLD = stage_policy["K_MATCH_DETACH_STAGE_THRESHOLD"]
    DETECT_ANOMALY = stage_policy["DETECT_ANOMALY"]

    ignored_train_keys, ignored_ngm_keys = _get_ignored_config_keys(train_config, ngm_config)
    if ignored_train_keys:
        print(f"[Stage {stage}] Ignoring train config keys: {ignored_train_keys}")
    if ignored_ngm_keys:
        print(f"[Stage {stage}] Ignoring ngm config keys: {ignored_ngm_keys}")

    print("BACKBONE_LR =", BACKBONE_LR)
    print("Start epoch: ", start_epoch)
    minimal_cfg = {
        "LR": LR,
        "K_LR": K_LR,
        "num_epochs": num_epochs,
        "num_iterations": num_iterations,
        "BATCH_SIZE": BATCH_SIZE,
        "BM_NAME": BM_NAME,
        "OVERFIT_TO_TRAIN_SPLIT": OVERFIT_TO_TRAIN_SPLIT,
        "OUTPUT_PATH": OUTPUT_PATH,
        "PRETRAINED_PATH": PRETRAINED_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "MATCH_BRANCH": match_branch,
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
        "K_CLS_RAMP": K_CLS_RAMP,
        "DUSTBIN_LOSS_WEIGHT": DUSTBIN_LOSS_WEIGHT,
        "SG_DUSTBIN_WEIGHT": SG_DUSTBIN_WEIGHT,
        "SG_DUSTBIN_RAMP": SG_DUSTBIN_RAMP,
        "DUSTBIN_REJECT_ENABLE": DUSTBIN_REJECT_ENABLE,
        "DUSTBIN_REJECT_MARGIN": DUSTBIN_REJECT_MARGIN,
        "AUTH_WEIGHT": AUTH_WEIGHT,
        "AUTH_WEIGHT_RAMP": AUTH_WEIGHT_RAMP,
        "AUTH_USE_POOLED": AUTH_USE_POOLED,
        "AUTH_SCALAR_DIM": AUTH_SCALAR_DIM,
        "K_MATCH_DETACH_STAGE_THRESHOLD": K_MATCH_DETACH_STAGE_THRESHOLD,
        "K_GATE_ENABLE": K_GATE_ENABLE,
        "K_GATE_THRESH": K_GATE_THRESH,
        "K_MATCH_ROUNDING": K_MATCH_ROUNDING,
        "AUTH_GATE_ENABLE": AUTH_GATE_ENABLE,
        "AUTH_GATE_THRESH": AUTH_GATE_THRESH,
        "MATCH_BRANCH": match_branch,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "WARMUP_K_EPOCHS": WARMUP_K_EPOCHS,
        "DETECT_ANOMALY": DETECT_ANOMALY,
        "K_GATE_THRESH_INIT": K_GATE_THRESH,
        "AUTH_GATE_THRESH_INIT": AUTH_GATE_THRESH,
        "DUSTBIN_REJECT_MARGIN_INIT": DUSTBIN_REJECT_MARGIN,
        "K_GATE_TEMP(HARDCODED)": K_GATE_TEMP,
        "K_GATE_LOSS_WEIGHT(HARDCODED)": K_GATE_LOSS_WEIGHT,
        "K_GATE_LEARN_STAGE_MIN(HARDCODED)": K_GATE_LEARN_STAGE_MIN,
        "AUTH_GATE_TEMP(HARDCODED)": AUTH_GATE_TEMP,
        "AUTH_GATE_LOSS_WEIGHT(HARDCODED)": AUTH_GATE_LOSS_WEIGHT,
        "DUSTBIN_MARGIN_TEMP(HARDCODED)": DUSTBIN_MARGIN_TEMP,
        "DUSTBIN_MARGIN_LOSS_WEIGHT(HARDCODED)": DUSTBIN_MARGIN_LOSS_WEIGHT,
        "IGNORED_TRAIN_KEYS": ignored_train_keys,
        "IGNORED_NGM_KEYS": ignored_ngm_keys,
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
        overfit_to_train_split=OVERFIT_TO_TRAIN_SPLIT,
    )
    # =====================================================
    # Model, Loss, and Device Setup
    # =====================================================
    model = Net(
        regression=REGRESSION,
        k_reg_weight=K_REG_WEIGHT,
        k_cls_weight=K_CLS_WEIGHT,
        dustbin_loss_weight=DUSTBIN_LOSS_WEIGHT,
    )
    model.apply_stage_policy(stage_policy)
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
    # Stage-wise progressive unfreezing
    # =====================================================
    stage_param_groups = _configure_stage_trainability(model, stage)
    optimizer, optimizer_k = _build_optimizers(model, stage_param_groups, LR, BACKBONE_LR, K_LR)

    stage_messages = {
        1: "Stage 1: train shared matcher baseline (matcher/backbone active; k/auth/dustbin frozen).",
        2: "Stage 2: train K + auth heads while matcher/backbone stay mostly frozen.",
        3: "Stage 3: joint top-k training with matcher/backbone/k/auth active.",
        4: "Stage 4: dustbin+auth warmup (matcher/backbone/K frozen).",
        5: "Stage 5: joint dustbin training with matcher/backbone/dustbin/auth active.",
    }
    print(stage_messages.get(stage, f"Stage {stage}: fallback full fine-tuning."))

    trainable_count = sum(int(p.requires_grad) for p in model.parameters())
    total_count = sum(1 for _ in model.parameters())
    print(f"Trainable params: {trainable_count}/{total_count}")



      

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
    warmup_epochs = WARMUP_EPOCHS
    warmup_k_epochs = WARMUP_K_EPOCHS
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
                
    

    # =====================================================
    # Training Loop
    # =====================================================
    for epoch in range(start_epoch, start_epoch + num_epochs):
        logger.info(f"Epoch {epoch}/{start_epoch + num_epochs - 1}")
        logger.info("-" * 50)
        print("Epoch {}/{}".format(epoch, start_epoch + num_epochs - 1))
        print("-" * 10)

        model.train()
        if stage in (1, 2, 3):
            # Stages with partial freezing should keep BN running stats fixed.
            model.apply(_set_batchnorm_eval)
        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))
        if optimizer_k is not None:
            print("K_regression_lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer_k.param_groups]))

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)

        if optimizer_k is not None:
            for i, param_group in enumerate(optimizer_k.param_groups):
                writer.add_scalar(f'Learning_Rate_K/group_{i}', param_group['lr'], epoch)

        denom = max(num_epochs - 1, 1)
        progress = max(0.0, min(1.0, (epoch - start_epoch) / denom))

        # Optional per-stage ramps.
        sg_weight = _ramp_value(SG_DUSTBIN_RAMP, progress, SG_DUSTBIN_WEIGHT)
        auth_weight_epoch = _ramp_value(AUTH_WEIGHT_RAMP, progress, AUTH_WEIGHT)
        model.k_cls_weight = _ramp_value(K_CLS_RAMP, progress, K_CLS_WEIGHT)

        writer.add_scalar('Train/SG_Dustbin_Weight', sg_weight, epoch)
        writer.add_scalar('Train/Auth_Weight', auth_weight_epoch, epoch)
        writer.add_scalar('Train/K_CLS_Weight', model.k_cls_weight, epoch)
        k_pred_ramp = progress if TRAIN_USE_PRED_K else 0.0
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
            auth_weight=auth_weight_epoch,
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
            auth_weight=auth_weight_epoch,
        )
    
        
        # Save best model based on validation loss and update checkpoint file
        if avg_val_total < best_loss:
            best_loss = avg_val_total
            no_improvement_count = 0
            best_model_path = str(checkpoint_path / "best_model.pt")
            save_model(model, best_model_path)
            with open(start_file, "w") as f:
                json.dump({"start_epoch": epoch + 1}, f)
            if stage in (4, 5):
                thresh_payload = {"auth_threshold": float(auth_threshold)}
                thresh_paths = [
                    checkpoint_path / "auth_threshold.json",
                    Path(OUTPUT_PATH) / "auth_threshold.json",
                ]
                for thresh_path in thresh_paths:
                    with open(thresh_path, "w") as f:
                        json.dump(thresh_payload, f)
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
