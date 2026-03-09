from __future__ import annotations

from pathlib import Path
import logging
import sys

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.benchmark import (
    L3SFBenchmark,
    L3SFV2AugmentedBenchmark,
    PolyUDBIBenchmark,
    PolyUDBIIBenchmark,
)
from src.gmdataset import GMDataset, RESCALE


DATASET_CHOICES = ["L3SFV2Augmented", "PolyU-DBII", "PolyU-DBI", "L3-SF"]


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
            force=True,
        )
    except (TypeError, ValueError):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
    return logging.getLogger(__name__)


def default_data_root(dataset_name: str) -> str:
    if dataset_name == "PolyU-DBII":
        return "dataset/PolyU/DBII"
    if dataset_name == "PolyU-DBI":
        return "dataset/PolyU/DBI"
    if dataset_name == "L3-SF":
        return "dataset/L3-SF"
    return "dataset/Synthetic"


def build_classify_dataset(
    dataset_name: str,
    data_root: str,
    *,
    filter=None,
    length=None,
    augment: bool = False,
) -> GMDataset:
    if dataset_name == "PolyU-DBII":
        benchmark = PolyUDBIIBenchmark(
            sets="test",
            obj_resize=RESCALE,
            train_root=data_root,
            task="classify",
            filter=filter,
        )
    elif dataset_name == "PolyU-DBI":
        benchmark = PolyUDBIBenchmark(
            sets="test",
            obj_resize=RESCALE,
            train_root=data_root,
            task="classify",
            filter=filter,
        )
    elif dataset_name == "L3-SF":
        benchmark = L3SFBenchmark(
            sets="test",
            obj_resize=RESCALE,
            train_root=data_root,
            task="classify",
            filter=filter,
        )
    else:
        dataset_name = "L3SFV2Augmented"
        benchmark = L3SFV2AugmentedBenchmark(
            sets="test",
            obj_resize=RESCALE,
            train_root=data_root,
            task="classify",
            name=dataset_name,
            filter=filter,
        )
    return GMDataset(dataset_name, benchmark, length, True, None, "2GM", augment=augment)


def compute_curve_stats(labels: np.ndarray, scores: np.ndarray, logger: logging.Logger | None = None):
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        if logger is not None:
            logger.warning("Only one class present in labels; ROC/PR/EER metrics are undefined.")
        return {
            "fpr": None,
            "tpr": None,
            "roc_auc": float("nan"),
            "prec_curve": None,
            "rec_curve": None,
            "pr_auc": float("nan"),
            "eer_threshold": None,
            "eer": float("nan"),
        }

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer_threshold = float(thresholds[eer_idx])
    eer = float((fpr[eer_idx] + fnr[eer_idx]) * 0.5)

    prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(auc(fpr, tpr)),
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "pr_auc": float(auc(rec_curve, prec_curve)),
        "eer_threshold": eer_threshold,
        "eer": eer,
    }
