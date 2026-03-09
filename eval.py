#!/usr/bin/env python3
"""Evaluate shared matches over a dataset split."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.gmdataset import get_dataloader
from src.model.dustbin import strip_dustbin_by_ns
from src.model.ngm import Net
from src.train.data_loader import build_dataloaders
from utils.eval_cli_common import compute_curve_stats
from utils.models_sl import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matcher over a split and report ROC + normalized/raw shared-k statistics."
        )
    )
    parser.add_argument("--config", default="stage3.yml", help="Stage config YAML for dataloader defaults.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split.")
    parser.add_argument("--dataset-len", type=int, default=640, help="Dataset length for GMDataset.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument(
        "--benchmark-name",
        choices=["L3SFV2AugmentedBenchmark", "L3SFBenchmark"],
        default=None,
        help="Override benchmark name from config.",
    )
    parser.add_argument("--train-root", default=None, help="Override dataset root.")
    parser.add_argument("--weights", default="results5/dustbin/stage3/params/best_model.pt", help="Checkpoint path.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Threshold for binary predictions.")
    parser.add_argument("--out-dir", default="debug_outputs/eval", help="Output directory.")
    return parser.parse_args()


def _default_train_root(benchmark_name: str) -> str:
    if benchmark_name == "L3SFBenchmark":
        return "dataset/L3-SF"
    return "dataset/Synthetic"


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return torch.device(device_arg)


def _to_1d_long_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.view(-1).to(device=device, dtype=torch.long)
    if isinstance(x, (list, tuple)):
        vals = []
        for item in x:
            if isinstance(item, torch.Tensor):
                vals.append(int(item.view(-1)[0].item()) if item.numel() > 0 else 0)
            else:
                vals.append(int(item))
        return torch.tensor(vals, device=device, dtype=torch.long)
    return torch.tensor([int(x)], device=device, dtype=torch.long)


def _move_to_device(value: Any, device: torch.device) -> Any:
    from src.sparse_torch.csx_matrix import CSRMatrix3d, CSCMatrix3d

    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (CSRMatrix3d, CSCMatrix3d)):
        if device.type == "cuda":
            return value.cuda()
        return value
    try:
        pyg_types = (pyg.data.Data, pyg.data.Batch, pyg.data.batch.DataBatch)
    except AttributeError:
        pyg_types = (pyg.data.Data, pyg.data.Batch)
    if isinstance(value, pyg_types):
        return value.to(device)
    return value


def _load_model(
    weights_path: Path,
    device: torch.device,
    ngm_cfg: Dict[str, Any],
) -> Net:
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = Net(
        regression=bool(ngm_cfg.get("REGRESSION", True)),
        dustbin_loss_weight=float(ngm_cfg.get("DUSTBIN_LOSS_WEIGHT", 0.5)),
    )
    model.train_use_pred_k = bool(ngm_cfg.get("TRAIN_USE_PRED_K", True))
    model.dustbin_reject_enable = bool(ngm_cfg.get("DUSTBIN_REJECT_ENABLE", True))
    model.dustbin_reject_margin = 0.0
    model.to(device)
    load_model(model, str(weights_path), strict=False)
    model.eval()
    return model


def _run_model(model: Net, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model_input = _move_to_device(copy.deepcopy(batch), device)
    with torch.no_grad():
        return model(model_input)


def _extract_real_perm(
    outputs: Dict[str, Any],
    *,
    perm_key: str = "perm_mat",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if perm_key not in outputs or "ns" not in outputs:
        raise KeyError(f"Model outputs missing required keys: {perm_key}, ns")

    perm = outputs[perm_key]
    if not isinstance(perm, torch.Tensor):
        perm = torch.as_tensor(perm)
    device = perm.device

    n1 = _to_1d_long_tensor(outputs["ns"][0], device)
    n2 = _to_1d_long_tensor(outputs["ns"][1], device)
    if bool(outputs.get("has_dustbin", False)):
        n1 = torch.clamp(n1 - 1, min=0)
        n2 = torch.clamp(n2 - 1, min=0)

    perm_real = strip_dustbin_by_ns(perm, n1, n2)
    if not isinstance(perm_real, torch.Tensor):
        perm_real = torch.as_tensor(perm_real, device=device)
    perm_bin = (perm_real > 0.5).to(torch.float32)
    return perm_bin, n1, n2


def _save_plots(
    out_dir: Path,
    labels: np.ndarray,
    raw_k: np.ndarray,
    norm_k: np.ndarray,
    curve_stats: Dict[str, Any],
) -> None:
    fpr = curve_stats["fpr"]
    tpr = curve_stats["tpr"]
    roc_auc = curve_stats["roc_auc"]
    prec_curve = curve_stats["prec_curve"]
    rec_curve = curve_stats["rec_curve"]
    pr_auc = curve_stats["pr_auc"]
    eer_threshold = curve_stats["eer_threshold"]

    if fpr is not None and tpr is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Shared Matches)")
        plt.legend(loc="lower right")
        plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    if rec_curve is not None and prec_curve is not None:
        plt.figure()
        plt.plot(rec_curve, prec_curve, label=f"PR AUC = {pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Shared Matches)")
        plt.legend(loc="lower left")
        plt.savefig(out_dir / "pr_curve.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    genuine_norm = norm_k[labels == 1]
    imposter_norm = norm_k[labels == 0]
    bins = np.linspace(0, 1, 30)
    plt.figure(figsize=(10, 6))
    plt.hist(imposter_norm, bins=bins, alpha=0.5, label="Imposter Matches", color="red")
    plt.hist(genuine_norm, bins=bins, alpha=0.5, label="Genuine Matches", color="green")
    plt.xlabel("Normalized k Value (shared_k/min_points)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Normalized k Values")
    plt.grid(alpha=0.3)
    if eer_threshold is not None:
        plt.axvline(x=eer_threshold, color="black", linestyle="--", label=f"EER Threshold ({eer_threshold:.3f})")
    plt.legend()
    plt.savefig(out_dir / "normalized_k_histogram.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    genuine_raw = raw_k[labels == 1]
    imposter_raw = raw_k[labels == 0]
    max_k = max(float(np.max(raw_k)) * 1.05, 1.0)
    bins = np.linspace(0, max_k, 30)
    plt.figure(figsize=(10, 6))
    plt.hist(imposter_raw, bins=bins, alpha=0.5, label="Imposter Matches", color="red")
    plt.hist(genuine_raw, bins=bins, alpha=0.5, label="Genuine Matches", color="green")
    plt.xlabel("Raw k Value (Number of Shared Matches)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Raw k Values")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "raw_k_histogram.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def _get_pair_ids(batch: Dict[str, Any], idx: int) -> tuple[str, str]:
    id_list = batch.get("id_list")
    if not isinstance(id_list, (list, tuple)) or len(id_list) < 2:
        return ("", "")
    try:
        return str(id_list[0][idx]), str(id_list[1][idx])
    except Exception:
        return ("", "")


def _evaluate_split(
    dataloader,
    model: Net,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    rows = []
    raw_k_list = []
    norm_k_list = []
    labels_list = []

    with torch.no_grad():
        for it, batch in enumerate(dataloader, start=1):
            outputs = _run_model(model, batch, device)
            shared_perm, n1, n2 = _extract_real_perm(outputs, perm_key="perm_mat")
            shared_k = shared_perm.sum(dim=(1, 2)).to(torch.float32).cpu()
            min_points = torch.minimum(n1, n2).to(torch.float32).cpu().clamp(min=1.0)
            norm_k = (shared_k / min_points).clamp(0.0, 1.0)

            labels = batch["label"].view(-1).detach().cpu().to(torch.int32)
            batch_size = int(labels.numel())

            raw_k_np = shared_k.numpy()
            norm_k_np = norm_k.numpy()
            labels_np = labels.numpy()

            raw_k_list.append(raw_k_np)
            norm_k_list.append(norm_k_np)
            labels_list.append(labels_np)

            for i in range(batch_size):
                id0, id1 = _get_pair_ids(batch, i)
                rows.append(
                    {
                        "id0": id0,
                        "id1": id1,
                        "label": int(labels_np[i]),
                        "shared_raw_k": float(raw_k_np[i]),
                        "shared_norm_k": float(norm_k_np[i]),
                        "min_points": float(min_points[i].item()),
                    }
                )

            if it % 10 == 0:
                print(f"Processed {it} batches...")

    if len(labels_list) == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "rows_df": pd.DataFrame(rows),
        "raw_k": np.concatenate(raw_k_list),
        "norm_k": np.concatenate(norm_k_list),
        "labels": np.concatenate(labels_list),
    }


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    ngm_cfg = cfg.get("ngm", {}) if isinstance(cfg, dict) else {}

    benchmark_name = args.benchmark_name or train_cfg.get("BM_NAME", "L3SFV2AugmentedBenchmark")
    train_root = args.train_root or _default_train_root(benchmark_name)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("BATCH_SIZE", 8))
    dataset_len = int(args.dataset_len)
    filter_value = train_cfg.get("FILTER", None)
    overfit = bool(train_cfg.get("OVERFIT_TO_TRAIN_SPLIT", False))

    train_loader, val_loader, test_loader = build_dataloaders(
        train_root=train_root,
        dataset_len=dataset_len,
        batch_size=batch_size,
        benchmark_name=benchmark_name,
        filter=filter_value,
        overfit_to_train_split=overfit,
    )

    if args.split == "train":
        dataloader = get_dataloader(train_loader.dataset, batch_size=batch_size, shuffle=False, fix_seed=True)
    elif args.split == "test":
        dataloader = test_loader
    else:
        dataloader = val_loader

    weights = Path(args.weights)
    requested_device = _resolve_device(args.device)

    model = _load_model(weights, requested_device, ngm_cfg)

    try:
        eval_data = _evaluate_split(dataloader, model, requested_device)
        actual_device = requested_device
    except RuntimeError as exc:
        if requested_device.type == "cuda" and "device-side assert" in str(exc):
            print("CUDA device-side assert detected; retrying full evaluation on CPU.")
            cpu_device = torch.device("cpu")
            model = _load_model(weights, cpu_device, ngm_cfg)
            eval_data = _evaluate_split(dataloader, model, cpu_device)
            actual_device = cpu_device
        else:
            raise

    labels = eval_data["labels"].astype(np.int32)
    raw_k = eval_data["raw_k"].astype(np.float32)
    norm_k = eval_data["norm_k"].astype(np.float32)

    curve_stats = compute_curve_stats(labels, norm_k)
    decision_threshold = float(args.decision_threshold)
    preds = (norm_k >= decision_threshold).astype(np.int32)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_data["rows_df"].to_csv(out_dir / "pair_scores.csv", index=False)

    metrics = {
        "split": args.split,
        "num_samples": int(labels.size),
        "num_genuine": int(np.sum(labels == 1)),
        "num_imposter": int(np.sum(labels == 0)),
        "decision_threshold": decision_threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(curve_stats["roc_auc"]),
        "pr_auc": float(curve_stats["pr_auc"]),
        "eer": float(curve_stats["eer"]),
        "eer_threshold": (
            float(curve_stats["eer_threshold"]) if curve_stats["eer_threshold"] is not None else float("nan")
        ),
        "far": float(far),
        "frr": float(frr),
        "device": str(actual_device),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    _save_plots(out_dir, labels, raw_k, norm_k, curve_stats)

    print("Evaluation complete.")
    print(f"Split: {args.split}")
    print(f"Samples: {metrics['num_samples']} (genuine={metrics['num_genuine']}, imposter={metrics['num_imposter']})")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
