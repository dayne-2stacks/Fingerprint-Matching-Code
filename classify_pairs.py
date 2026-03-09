#!/usr/bin/env python
"""Classify fingerprint pairs using k_ratio and dustbin rejection.

This script loads a trained model, runs it over dataset-generated pairs,
computes k_ratio and a dustbin unmatched rate, applies thresholds to
label genuine vs imposter, and writes per-pair predictions to CSV.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import torch

from src.gmdataset import GMDataset, get_dataloader
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from utils.eval_cli_common import (
    DATASET_CHOICES,
    build_classify_dataset,
    default_data_root,
    setup_logging,
)
from utils.models_sl import load_model
from src.model.dustbin import strip_dustbin_from_outputs


def _collect_pairs(
    model: Net,
    dataset: GMDataset,
    device: torch.device,
    max_pairs: int | None,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=False, fix_seed=True)

    rows: List[dict] = []
    all_scores: List[float] = []
    all_labels: List[int] = []

    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = data_to_cuda(batch)
            outputs = model(batch)
            dustbin_rates = None
            # Do if dustbin rejection is enabled, compute dustbin unmatched rate
            if outputs.get("has_dustbin", False):
                # Perm matrix from model output
                perm_full = outputs["perm_mat"].detach()
                # number of points including dustbin
                ns = batch["ns"]
                #  detach the list of ns and move to cpu
                n1_list = ns[0].detach().cpu().tolist()
                n2_list = ns[1].detach().cpu().tolist()
                # 
                dustbin_rates = []
                # maximum number of rows/cols in the perm matrix
                max_rows = perm_full.shape[1]
                max_cols = perm_full.shape[2]
                #  Loop through the matrix
                for i in range(perm_full.shape[0]):
                    n1 = int(n1_list[i])
                    n2 = int(n2_list[i])
                    # there are no points in both fingerprints, continue
                    if n1 == 0 and n2 == 0:
                        dustbin_rates.append(0.0)
                        continue
                    # Resolve dustbin row/col indices based on actual tensor shape.
                    #  If n1 or n2 exceed max_rows/max_cols, clamp to max-1 (last index)
                    dust_row = min(n1, max_rows - 1)
                    dust_col = min(n2, max_cols - 1)

                    # Count number of points excluding dustbin
                    row_count = n1 if dust_row == n1 else dust_row
                    col_count = n2 if dust_col == n2 else dust_col
                    # ensure non-negative
                    row_count = max(row_count, 0)
                    col_count = max(col_count, 0)
                    # Sum all the dustbin rows and columns to get the total unmatched count in the dustbin
                    row_dust = perm_full[i, :row_count, dust_col].sum()
                    col_dust = perm_full[i, dust_row, :col_count].sum()
                    dustbin_unmatched = row_dust + col_dust
                    denom = max(n1 + n2, 1)
                    dustbin_rates.append(float(dustbin_unmatched.item()) / float(denom))
                # strip the dustbin row/col from the perm matrix and update ns to reflect the new size
                strip_dustbin_from_outputs(outputs)

            # detach the permutation matrix why do we 
            perm_mat = outputs["perm_mat"].detach()
            k_pred = perm_mat.sum(dim=(1, 2)).float()

            ns = batch["ns"]
            min_points = torch.min(ns[0], ns[1]).float().clamp(min=1.0)
            k_ratio = (k_pred / min_points).clamp(0, 1)
            scores = k_ratio

            labels = batch["label"].view(-1).detach().cpu().numpy().astype(int)
            scores_np = scores.detach().cpu().numpy().astype(float)
            k_pred_np = k_pred.detach().cpu().numpy().astype(float)
            min_points_np = min_points.detach().cpu().numpy().astype(float)
            if dustbin_rates is None:
                dustbin_rates = [0.0] * len(labels)

            id_list = batch["id_list"]
            batch_size = len(labels)

            for i in range(batch_size):
                id0 = id_list[0][i]
                id1 = id_list[1][i]
                rows.append(
                    {
                        "id0": id0,
                        "id1": id1,
                        "label": int(labels[i]),
                        "score": float(scores_np[i]),
                        "raw_k": float(k_pred_np[i]),
                        "min_points": float(min_points_np[i]),
                        "k_ratio": float(scores_np[i]),
                        "dustbin_rate": float(dustbin_rates[i]),
                    }
                )
                all_labels.append(int(labels[i]))
                all_scores.append(float(scores_np[i]))
                processed += 1

                if max_pairs is not None and processed >= max_pairs:
                    logger.info("Reached max pairs limit: %d", max_pairs)
                    df = pd.DataFrame(rows)
                    return df, np.array(all_labels), np.array(all_scores)

    df = pd.DataFrame(rows)
    return df, np.array(all_labels), np.array(all_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify fingerprint pairs using k_ratio and dustbin rate.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="L3SFV2Augmented",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory of the dataset. If omitted a sensible default is used.",
    )
    parser.add_argument(
        "--model-path",
        default="results/base_joint_ft/params/best_model.pt",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Deprecated alias for --k-min.",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=0.5,
        help="Minimum k_ratio to accept a match.",
    )
    parser.add_argument(
        "--dustbin-max",
        type=float,
        default=0.5,
        help="Maximum dustbin_rate to accept a match.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to process.",
    )
    parser.add_argument(
        "--out",
        default="results/base_joint_ft/pair_predictions.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    logger = setup_logging()

    data_root = args.data_root or default_data_root(args.dataset)
    logger.info("Dataset=%s data_root=%s", args.dataset, data_root)

    dataset = build_classify_dataset(args.dataset, data_root)

    model = Net(regression=True)
    model.dustbin_reject_enable = True
    model.dustbin_reject_margin = 0.2

    model_path = Path(args.model_path)
    if model_path.exists():
        load_model(model, str(model_path))
        logger.info("Loaded model checkpoint: %s", model_path)
    else:
        logger.warning("Checkpoint not found: %s (running with random init)", model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    logger.info("Device: %s", device)

    df, labels, scores = _collect_pairs(
        model=model,
        dataset=dataset,
        device=device,
        max_pairs=args.max_pairs,
        logger=logger,
    )

    unique_labels = np.unique(labels) if labels.size > 0 else np.array([])
    if args.threshold is not None:
        logger.warning("--threshold is deprecated; use --k-min instead.")
        k_min = float(args.threshold)
    else:
        k_min = float(args.k_min)
    dustbin_max = float(args.dustbin_max)

    dustbin_rates = df["dustbin_rate"].to_numpy(dtype=float) if "dustbin_rate" in df else np.zeros_like(scores)
    preds = ((scores >= k_min) & (dustbin_rates <= dustbin_max)).astype(int)

    df["pred"] = preds
    df["k_min"] = k_min
    df["dustbin_max"] = dustbin_max

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary metrics
    genuine_count = int(np.sum(labels == 1)) if labels.size > 0 else 0
    imposter_count = int(np.sum(labels == 0)) if labels.size > 0 else 0

    logger.info("k_min used: %.4f", k_min)
    logger.info("dustbin_max used: %.4f", dustbin_max)
    logger.info("Genuine pairs: %d", genuine_count)
    logger.info("Imposter pairs: %d", imposter_count)

    if unique_labels.size >= 2:
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        logger.info("Accuracy: %.4f", accuracy)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall: %.4f", recall)
        logger.info("F1: %.4f", f1)
        logger.info("FAR: %.4f", far)
        logger.info("FRR: %.4f", frr)
    else:
        logger.warning("Metrics skipped: only one class present in labels.")

    print(f"Saved predictions to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
