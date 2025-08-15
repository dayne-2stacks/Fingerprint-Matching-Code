"""Evaluate the trained binary classifier and report common metrics.

The script loads the best model from ``results/binary-classifier/params``
and computes several verification metrics on the test split of either the
L3SFV2Augmented or PolyU DBII dataset.  The computed metrics are saved in
``metrics.csv`` and ROC/PR curves are written as PNG files in the same
directory.
"""

from pathlib import Path
import argparse
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
from utils.matching import build_matches

from src.benchmark import L3SFV2AugmentedBenchmark, PolyUDBIIBenchmark, PolyUDBIBenchmark, L3SFBenchmark
from src.gmdataset import GMDataset, get_dataloader, TestDataset
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import load_model
from utils.visualize import visualize_stochastic_matrix, visualize_match, to_grayscale_cv2_image


def evaluate(dataset_name: str, data_root: str):
    """Run evaluation using the best classifier model for the chosen dataset.

    Metrics are written to ``metrics.csv`` and also logged to ``eval.log``
    inside ``results/binary-classifier``.
    """
    dataset_len = None

    if dataset_name == "PolyU-DBII":
        benchmark = PolyUDBIIBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
        )
    elif dataset_name == "PolyU-DBI":
        benchmark = PolyUDBIBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
        )
    elif dataset_name == "L3-SF":
        benchmark = L3SFBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
        )
        
    else:
        benchmark = L3SFV2AugmentedBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
        )
        dataset_name = "L3SFV2Augmented"

    dataset = TestDataset(dataset_name, benchmark, dataset_len, True, None, "2GM", augment=False)
    dataloader = get_dataloader(dataset, shuffle=True, fix_seed=True)

    match_net = Net(regression=True)

    # Load the best weights of the classifier model. The network outputs a
    # match probability that combines both the predicted number of
    # correspondences and the underlying similarity scores.
    model_path = Path("results/binary-classifier/params/best_model.pt")

    if model_path.exists():
        load_model(match_net, str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match_net.to(device).eval()

    all_labels = []
    all_probs = []
    iteration = 0
    with torch.no_grad():
        for batch in dataloader:
            iteration += 1
            
            batch = data_to_cuda(batch)
            outputs = match_net(batch)
            if "cls_prob" in outputs:
                prob = outputs["cls_prob"].detach()
            else:
                # Fallback to using the ratio of predicted correspondences
                perm_mat = outputs["perm_mat"].detach()
                k_pred = perm_mat.sum(dim=(1, 2)).float()
                ns = batch["ns"]
                min_points = torch.min(ns[0], ns[1]).float()
                prob = (k_pred / min_points).clamp(0, 1)
            all_probs.append(prob.cpu())
            all_labels.append(batch["label"].cpu())
            if iteration % 5 == 0:
                print(f"Processed {iteration} batches...")
                

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Debug: Check label distribution
    print(f"Total samples: {len(all_labels)}")
    print(f"Genuine matches (label=1): {np.sum(all_labels == 1)}")
    print(f"Imposter matches (label=0): {np.sum(all_labels == 0)}")
    print(f"Unique labels: {np.unique(all_labels)}")
    
    # If no genuine matches, let's check the first few batches manually
    if np.sum(all_labels == 1) == 0:
        print("No genuine matches found! Checking first few batches...")
        debug_dataloader = get_dataloader(dataset, shuffle=False, fix_seed=True)
        for i, batch in enumerate(debug_dataloader):
            if i >= 5:  # Check first 5 batches
                break
            batch = data_to_cuda(batch)
            labels = batch["label"].cpu().numpy()
            print(f"Batch {i}: labels = {labels}")
                

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

    out_dir = Path(f"results/binary-classifier/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize one genuine match (label == 1) from the network
    for i, (label, prob) in enumerate(zip(all_labels, all_probs)):
        if label == 1:
            # Re-run the dataloader to get the corresponding batch and visualize
            count = 0
            for batch in get_dataloader(dataset, shuffle=False, fix_seed=True):
                batch = data_to_cuda(batch)
                batch_label = batch["label"].cpu().numpy()[0]
                if batch_label == 1:
                    # Run the model to get outputs
                    with torch.no_grad():
                        outputs = match_net(batch)
                    
                    # Get keypoints
                    if 'Ps' in batch:
                        kp0 = batch['Ps'][0][0].cpu().numpy()
                        kp1 = batch['Ps'][1][0].cpu().numpy()
                    else:
                        # Fallback keypoints if not available
                        kp0 = np.array([[100, 100], [150, 150], [200, 200]])
                        kp1 = np.array([[110, 110], [160, 160], [210, 210]])
                    
                    # Get images
                    if "images" in batch:
                        img0 = batch["images"][0][0]
                        img1 = batch["images"][1][0]
                        img0 = to_grayscale_cv2_image(img0)
                        img1 = to_grayscale_cv2_image(img1)
                    else:
                        # Create placeholder images if not available
                        img0 = np.zeros((240, 320), dtype=np.uint8)
                        img1 = np.zeros((240, 320), dtype=np.uint8)
                    
                    # Get matching matrices
                    ds_mat = outputs["ds_mat"].cpu().numpy()[0]
                    per_mat = outputs["perm_mat"].cpu().numpy()[0]
                    
                    # Build matches using the same function as train.py
                    matches = build_matches(ds_mat, per_mat)
                    
                    # Visualize matches using the same function as train.py
                    visualize_match(img0, img1, kp0, kp1, matches, 
                                  prefix=str(out_dir) + "/", 
                                  filename="genuine_match_example")
                    
                    # Also visualize the stochastic matrix

                    
                    print(f"Genuine match visualization saved with {len(matches)} matches")
                    print(f"Probability: {prob:.4f}")
                    break
                count += 1
            break
    
    # Do the same for an imposter match (label == 0)
    for i, (label, prob) in enumerate(zip(all_labels, all_probs)):
        if label == 0:
            count = 0
            for batch in get_dataloader(dataset, shuffle=False, fix_seed=True):
                batch = data_to_cuda(batch)
                batch_label = batch["label"].cpu().numpy()[0]
                if batch_label == 0:
                    with torch.no_grad():
                        outputs = match_net(batch)
                    
                    if 'Ps' in batch:
                        kp0 = batch['Ps'][0][0].cpu().numpy()
                        kp1 = batch['Ps'][1][0].cpu().numpy()
                    else:
                        kp0 = np.array([[100, 100], [150, 150], [200, 200]])
                        kp1 = np.array([[110, 110], [160, 160], [210, 210]])
                    
                    if "images" in batch:
                        img0 = batch["images"][0][0]
                        img1 = batch["images"][1][0]
                        img0 = to_grayscale_cv2_image(img0)
                        img1 = to_grayscale_cv2_image(img1)
                    else:
                        img0 = np.zeros((240, 320), dtype=np.uint8)
                        img1 = np.zeros((240, 320), dtype=np.uint8)
                    
                    ds_mat = outputs["ds_mat"].cpu().numpy()[0]
                    per_mat = outputs["perm_mat"].cpu().numpy()[0]
                    matches = build_matches(ds_mat, per_mat)
                    
                    visualize_match(img0, img1, kp0, kp1, matches, 
                                  prefix=str(out_dir) + "/", 
                                  filename="imposter_match_example")
                    
           
                    
                    print(f"Imposter match visualization saved with {len(matches)} matches")
                    print(f"Probability: {prob:.4f}")
                    break
                count += 1
            break

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
    parser = argparse.ArgumentParser(description="Evaluate the trained binary classifier")
    parser.add_argument(
        "--dataset",
        choices=["L3SFV2Augmented", "PolyU-DBII", "PolyU-DBI", "L3-SF"],
        default="L3-SF",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory of the dataset. If omitted a sensible default is used.",
    )
    args = parser.parse_args()

    if args.data_root is None:
        if args.dataset == "PolyU-DBII":
            data_root = "dataset/PolyU/DBII"
        elif args.dataset == "PolyU-DBI":
            data_root = "dataset/PolyU/DBI"
        elif args.dataset == "L3-SF":
            data_root = "dataset/L3-SF"
        else:
            data_root = "dataset/Synthetic"
    else:
        data_root = args.data_root

    evaluate(args.dataset, data_root)
