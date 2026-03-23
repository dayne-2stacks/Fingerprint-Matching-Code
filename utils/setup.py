from pathlib import Path
import yaml
import torch.optim as optim

def _stage_from_filename(name: str) -> int:
    name = name.lower()
    if "stage0" in name:
        return 0
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
    if stage == 0:
        return "matcher_warmup"
    if stage == 1:
        return "shared_matcher"
    if stage == 2:
        return "dustbin+topk"
    if stage == 3:
        return "dustbin"
    if stage == 4:
        return "joint"
    return "full"



def _load_global_config(cfg_file):
    with open(cfg_file, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    return {
        "train_defaults": raw_cfg.get("train_defaults", {}),
        "policy_defaults": raw_cfg.get("policy_defaults", {}),
        "data_defaults": raw_cfg.get("data_defaults", {}),
        "model_defaults": raw_cfg.get("model_defaults", {}),
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
        # Strip DDP "module." prefix so name matching works regardless of wrapping.
        n = name[len("module."):] if name.startswith("module.") else name
        if n.startswith(("encoder_k.", "final_row.", "final_col.")):
            groups["k_head"].append(param)
        elif n == "bin_score":
            groups["dustbin"].append(param)
        elif n.startswith("node_layers."):
            groups["backbone_node"].append(param)
        elif n.startswith(("edge_layers.", "final_layers.")):
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

    if stage == 0:
        # Stage 0: matcher warmup only, genuine pairs.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["backbone_node"], False)
        _set_trainable(groups["backbone_edge"], False)
        _set_trainable(groups["k_head"], False)
        _set_trainable(groups["dustbin"], False)
    elif stage == 1:
        # Stage 1: matcher + backbone, genuine pairs.
        _set_trainable(groups["matcher"], True)
        _set_trainable(groups["backbone_node"], True)
        _set_trainable(groups["backbone_edge"], True)
        _set_trainable(groups["k_head"], False)
        _set_trainable(groups["dustbin"], False)
    elif stage == 2:
        # Stage 2: dustbin + k-head together, genuine + imposter pairs.
        # Matcher/backbone stay frozen so both modules learn against a stable Sinkhorn.
        _set_trainable(groups["matcher"], False)
        _set_trainable(groups["backbone_node"], False)
        _set_trainable(groups["backbone_edge"], False)
        _set_trainable(groups["k_head"], True)
        _set_trainable(groups["dustbin"], True)
    elif stage == 3:
        # Stage 3: k-regression + dustbin, genuine + imposter pairs.
        # Dustbin continues refining while k-head learns against it.
        _set_trainable(groups["matcher"], False)
        _set_trainable(groups["backbone_node"], False)
        _set_trainable(groups["backbone_edge"], False)
        _set_trainable(groups["k_head"], True)
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
        1: 0,
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
