#!/usr/bin/env python3
"""Run a single forward pass and print match-matrix diagnostics.

This script loads a model checkpoint, pulls one sample from the requested
split, and prints ds_mat/per_mat stats used to understand fan-out matches.
"""
import argparse
from pathlib import Path
import cv2
import yaml
import numpy as np
import torch

from src.train.data_loader import build_dataloaders
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import load_model
from src.model.dustbin import strip_dustbin_by_ns
from utils.matching import build_matches
from utils.visualize import visualize_match, to_grayscale_cv2_image
from src.gmdataset import _standardize


def parse_args():
    parser = argparse.ArgumentParser(description="Debug final match matrices without training")
    parser.add_argument("--config", default="stage4.yml", help="Path to stage config YAML")
    parser.add_argument("--weights", default="results1/base_joint_ft/params/best_model.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to sample")
    parser.add_argument("--dataset-len", type=int, default=640, help="Dataset length for GMDataset")
    parser.add_argument("--image-source", choices=["raw", "sample"], default="raw", help="Image source for visualization")
    parser.add_argument("--pair-type", choices=["any", "genuine", "imposter"], default="any",
                        help="Select a genuine or imposter pair when sampling")
    parser.add_argument("--out-dir", default="debug_outputs", help="Output directory for saved images")
    parser.add_argument("--filename", default="final_match-debug", help="Base filename for match visualization")
    return parser.parse_args()

def _infer_batch_size(batch):
    if isinstance(batch, dict):
        if "batch_size" in batch:
            try:
                return int(batch["batch_size"])
            except (TypeError, ValueError):
                pass
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return int(v.shape[0])
            if isinstance(v, list) and v and isinstance(v[0], torch.Tensor) and v[0].dim() > 0:
                return int(v[0].shape[0])
    return 1


def _slice_value(value, idx, batch_size):
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value
        return value[idx:idx + 1]
    if isinstance(value, list):
        if value and all(isinstance(x, torch.Tensor) for x in value):
            if len(value) == batch_size:
                return value[idx]
            return [x[idx:idx + 1] if x.dim() > 0 else x for x in value]
        if len(value) == batch_size:
            return value[idx]
        return value
    if isinstance(value, dict):
        return {k: _slice_value(v, idx, batch_size) for k, v in value.items()}
    return value


def _slice_batch(batch, idx):
    batch_size = _infer_batch_size(batch)
    if isinstance(batch, dict):
        return {k: _slice_value(v, idx, batch_size) for k, v in batch.items()}
    return batch


def _extract_label(sample):
    if "label" not in sample:
        return None
    label = sample["label"]
    if isinstance(label, torch.Tensor):
        if label.numel() > 0:
            return float(label.flatten()[0].item())
        return None
    try:
        return float(label)
    except (TypeError, ValueError):
        return None


def _find_pair_index(batch, pair_type):
    if pair_type == "any":
        return 0
    if "label" not in batch:
        print("[WARN] No label in batch; falling back to first sample.")
        return 0
    labels = batch["label"]
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().float().flatten().cpu().numpy()
    else:
        labels = np.array(labels, dtype=np.float32).flatten()
    if pair_type == "genuine":
        matches = np.where(labels >= 0.5)[0]
    else:
        matches = np.where(labels < 0.5)[0]
    return int(matches[0]) if matches.size > 0 else None


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_config = config.get("train", {})
    ngm_config = config.get("ngm", {})

    dataset_len = int(args.dataset_len)
    train_root = "dataset/Synthetic"
    BM_NAME = train_config.get("BM_NAME", "L3SFV2AugmentedBenchmark")
    FILTER = train_config.get("FILTER", None)
    batch_size = train_config.get("BATCH_SIZE", 8)

    dataloader, val_dataloader, test_dataloader = build_dataloaders(
        train_root,
        dataset_len,
        batch_size=batch_size,
        benchmark_name=BM_NAME,
        filter=FILTER,
    )

    if args.split == "train":
        sample_loader = dataloader
    elif args.split == "test":
        sample_loader = test_dataloader
    else:
        sample_loader = val_dataloader

    # Model config mirrors train.py
    model = Net(
        regression=ngm_config.get("REGRESSION", True),
        k_reg_weight=ngm_config.get("K_REG_WEIGHT", 0.2),
        k_cls_weight=ngm_config.get("K_CLS_WEIGHT", 1.0),
        dustbin_loss_weight=ngm_config.get("DUSTBIN_LOSS_WEIGHT", 0.5),
    )
    model.dustbin_reject_enable = bool(train_config.get("DUSTBIN_REJECT_ENABLE", True))
    model.dustbin_reject_margin = float(train_config.get("DUSTBIN_REJECT_MARGIN", 0.0))
    model.train_use_pred_k = bool(ngm_config.get("TRAIN_USE_PRED_K", False))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Resolve checkpoint path
    weight_path = args.weights or train_config.get("PRETRAINED_PATH", "")
    if not weight_path:
        checkpoint_root = Path(train_config.get("CHECKPOINT_PATH", "checkpoints"))
        weight_path = str(checkpoint_root / "best_model.pt")

    if not Path(weight_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    print(f"Loading model weights from: {weight_path}")
    load_model(model, weight_path, strict=False)

    model.eval()
    single_sample = None
    outputs = None
    selected_label = None
    for batch in sample_loader:
        batch = data_to_cuda(batch)
        with torch.no_grad():
            batch_outputs = model(batch)
        idx = _find_pair_index(batch, args.pair_type)
        if idx is None:
            continue
        single_sample = _slice_batch(batch, idx)
        outputs = _slice_batch(batch_outputs, idx)
        selected_label = _extract_label(single_sample)
        if args.pair_type != "any":
            print(f"[INFO] Selected {args.pair_type} pair (label={selected_label}).")
        break

    if single_sample is None or outputs is None:
        raise RuntimeError(f"No {args.pair_type} pair found in {args.split} split.")


    print("has_dustbin:", outputs.get("has_dustbin"))
    print("perm_mat shape before:", outputs["perm_mat"].shape)
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
    print("perm_mat shape after:", outputs["perm_mat"].shape)

    if "Ps" in single_sample:
        kp0 = single_sample["Ps"][0][0].cpu().numpy()
        kp1 = single_sample["Ps"][1][0].cpu().numpy()
    else:
        raise KeyError("Sample does not contain 'Ps' keypoints; cannot compute kp stats")

    ds_mat = outputs["ds_mat"].cpu().numpy()[0]
    per_mat = outputs["perm_mat"].cpu().numpy()[0]
    gt_per_mat = outputs["gt_perm_mat"].cpu().numpy()[0]
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

    label_value = _extract_label(single_sample)

    print("ds_mat shape:", ds_mat.shape)
    print("per_mat shape:", per_mat.shape)
    print("kp0:", kp0.shape, "kp1:", kp1.shape)

    row_sum = per_mat.sum(axis=1)
    col_sum = per_mat.sum(axis=0)
    print("perm row sum stats:", row_sum.min(), row_sum.max(), row_sum.mean())
    print("perm col sum stats:", col_sum.min(), col_sum.max(), col_sum.mean())
    print("perm total matches:", int(per_mat.sum()))

    if "rejected_rows" in outputs:
        rr = outputs["rejected_rows"]
        if isinstance(rr, list) and rr and isinstance(rr[0], torch.Tensor):
            rr0 = rr[0]
            if rr0.numel() > 0:
                print("rejected_rows count:", int(rr0.sum().item()), "/", int(rr0.numel()))
            else:
                print("rejected_rows count: 0 / 0")
    if "rejected_cols" in outputs:
        rc = outputs["rejected_cols"]
        if isinstance(rc, list) and rc and isinstance(rc[0], torch.Tensor):
            rc0 = rc[0]
            if rc0.numel() > 0:
                print("rejected_cols count:", int(rc0.sum().item()), "/", int(rc0.numel()))
            else:
                print("rejected_cols count: 0 / 0")

    if "k_prob" in outputs:
        k_prob = outputs["k_prob"]
        if isinstance(k_prob, torch.Tensor):
            k_prob = float(k_prob.view(-1)[0].item())
        print("k_prob:", k_prob)
    if "k_logit" in outputs:
        k_logit = outputs["k_logit"]
        if isinstance(k_logit, torch.Tensor):
            k_logit = float(k_logit.view(-1)[0].item())
        print("k_logit:", k_logit)
    if "ns" in outputs:
        if isinstance(outputs["ns"], (list, tuple)) and len(outputs["ns"]) == 2:
            n1_dbg = outputs["ns"][0]
            n2_dbg = outputs["ns"][1]
            if isinstance(n1_dbg, torch.Tensor):
                n1_dbg = int(n1_dbg[0].item()) if n1_dbg.dim() > 0 else int(n1_dbg.item())
            if isinstance(n2_dbg, torch.Tensor):
                n2_dbg = int(n2_dbg[0].item()) if n2_dbg.dim() > 0 else int(n2_dbg.item())
            min_points_dbg = max(min(n1_dbg, n2_dbg), 1)
            print("min_points:", min_points_dbg)
            if "k_prob" in outputs:
                try:
                    k_match_dbg = float(k_prob) * float(min_points_dbg)
                    print("k_match (pred):", k_match_dbg)
                except Exception:
                    pass

    best_cols = np.argmax(ds_mat, axis=1)
    bincount = np.bincount(best_cols, minlength=ds_mat.shape[1])
    top5 = bincount.argsort()[-5:][::-1]
    print("top-5 most common cols:", top5)


    print("gt_perm_mat stats:",
      "min", gt_per_mat.min(),
      "max", gt_per_mat.max(),
      "sum", gt_per_mat.sum(),
      "row sums min/max", gt_per_mat.sum(axis=1).min(), gt_per_mat.sum(axis=1).max())


    match_perm = per_mat
    # if label_value is not None and label_value < 0.5:
    #     match_perm = gt_per_mat
    matches = build_matches(ds_mat, match_perm)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{out_dir}/"

    img0 = None
    img1 = None
    if args.image_source == "raw":
        if "id_list" not in single_sample:
            print("[WARN] id_list missing; falling back to sample images.")
        elif getattr(sample_loader.dataset, "augment", False):
            print("[WARN] Dataset augmentation enabled; raw images won't align. Falling back to sample images.")
        else:
            id_list = single_sample["id_list"]
            if isinstance(id_list, (list, tuple)) and id_list and isinstance(id_list[0], (list, tuple)):
                id_list = id_list[0]
            if len(id_list) < 1:
                print("[WARN] id_list is empty; falling back to sample images.")
            else:
                img_path0 = sample_loader.dataset.bm.get_path(id_list[0])
                img_path1 = None
                is_genuine = label_value is not None and label_value >= 0.5
                if is_genuine:
                    if getattr(sample_loader.dataset, "augment", False):
                        print("[WARN] Genuine pair with augmentation; raw image won't align. Falling back to sample images.")
                    else:
                        img_path1 = img_path0
                        print("[INFO] Genuine pair uses the same source image; reusing image0 for image1.")
                else:
                    if len(id_list) < 2:
                        print("[WARN] id_list has fewer than 2 entries; falling back to sample images.")
                    else:
                        img_path1 = sample_loader.dataset.bm.get_path(id_list[1])

                if img_path1 is None:
                    img0 = None
                    img1 = None
                    # Will fall back to sample images below.
                else:
                    raw0 = cv2.imread(img_path0)
                    raw1 = cv2.imread(img_path1)
                    if raw0 is None or raw1 is None:
                        print("[WARN] Failed to read raw images; falling back to sample images.")
                    else:
                        img0, _ = _standardize(raw0, [])
                        img1, _ = _standardize(raw1, [])
                        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if img0 is None or img1 is None:
        img0 = single_sample["images"][0][0]
        img1 = single_sample["images"][1][0]
        img0 = to_grayscale_cv2_image(img0)
        img1 = to_grayscale_cv2_image(img1)

    visualize_match(img0, img1, kp0, kp1, matches, prefix=prefix, filename=args.filename)
    print(f"Saved match visualization to {prefix}{args.filename}.jpg")

if __name__ == "__main__":
    main()
