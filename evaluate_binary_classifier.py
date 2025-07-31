"""Evaluate the trained binary classifier and report common metrics.

The script loads the best model from ``results/binary-classifier/params``
and computes several verification metrics on the test split.  The computed
metrics are saved in ``metrics.csv`` and ROC/PR curves are written as PNG
files in the same directory.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)
import torch

from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import load_model


def evaluate():
    """Run evaluation using the best classifier model.

    Metrics are written to ``metrics.csv`` and also logged to ``eval.log``
    inside ``results/binary-classifier``.
    """
    dataset_len = 640
    data_root = "dataset/Synthetic"

    benchmark = L3SFV2AugmentedBenchmark(
        sets="test",
        obj_resize=(320, 240),
        train_root=data_root,
        task="classify",
    )
    dataset = GMDataset("L3SFV2Augmented", benchmark, dataset_len, True, None, "2GM", augment=False)
    dataloader = get_dataloader(dataset, shuffle=False, fix_seed=True)

    match_net = Net(regression=True)

    # Load the best weights of the classifier model. The predicted k
    # value from the network determines whether two fingerprints match.
    model_path = Path("results/binary-classifier/params/best_model.pt")

    if model_path.exists():
        load_model(match_net, str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match_net.to(device).eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            batch = data_to_cuda(batch)
            outputs = match_net(batch)
            perm_mat = outputs["perm_mat"].detach()
            k_pred = perm_mat.sum(dim=(1, 2)).float()
            ns = batch["ns"]
            min_points = torch.min(ns[0], ns[1]).float()
            prob = (k_pred / min_points).clamp(0, 1)
            all_probs.append(prob.cpu())
            all_labels.append(batch["label"].cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_idx]
    preds = (all_probs >= eer_threshold).astype(np.int32)

    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    roc_auc = auc(fpr, tpr)

    prec_curve, rec_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec_curve, prec_curve)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    out_dir = Path("results/binary-classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(out_dir / "eval.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # PR curve
    plt.figure()
    plt.plot(rec_curve, prec_curve, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(out_dir / "pr_curve.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "far": far,
        "frr": frr,
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        logger.info("%s: %.4f", k, v)


if __name__ == "__main__":
    evaluate()
