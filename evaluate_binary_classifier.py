from pathlib import Path
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
from src.gmdataset import GMDataset, get_dataloader
from src.model.ngm import Net
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import load_model
from utils.visualize import visualize_stochastic_matrix, visualize_match, to_grayscale_cv2_image
from src.model.dustbin import strip_dustbin_by_ns


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(str(log_path)),
    ]
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
            force=True,
        )
    except (TypeError, ValueError):
        # Python < 3.8 doesn't support force=
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )
    return logging.getLogger(__name__)


def evaluate(dataset_name: str, data_root: str, filter=None):
    """Run evaluation using the best classifier model for the chosen dataset.
    """
    dataset_len = None

    if dataset_name == "PolyU-DBII":
        benchmark = PolyUDBIIBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
            filter=filter,
        )
    elif dataset_name == "PolyU-DBI":
        benchmark = PolyUDBIBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
            filter=filter,
        )
    elif dataset_name == "L3-SF":
        benchmark = L3SFBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
            filter=filter,
        )
        
    else:
        dataset_name = "L3SFV2Augmented"
        benchmark = L3SFV2AugmentedBenchmark(
            sets="test",
            obj_resize=(320, 240),
            train_root=data_root,
            task="classify",
            name =dataset_name,
            filter=filter,
        )
    
    # Path to model
    main_dir = "results/base_joint_ft"
    # Output directory for evaluation results
    out_dir = Path(f"{main_dir}/{dataset_name}")

    logger = _setup_logging(out_dir / "eval.log")
    logger.info("Starting evaluation: dataset=%s data_root=%s", dataset_name, data_root)

    # Create dataset and dataloader
    dataset = GMDataset(dataset_name, benchmark, dataset_len, True, None, "2GM", augment=False)
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=True, fix_seed=True)

    # Load the trained model
    match_net = Net(regression=True)
    checkpoint_root = Path(main_dir)
    model_path = checkpoint_root / "params" / "best_model.pt"

    if model_path.exists():
        load_model(match_net, str(model_path))
        logger.info("Loaded model checkpoint: %s", model_path)
    else:
        logger.warning("Checkpoint not found: %s (running with random init)", model_path)

    # Set up device and model for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match_net.dustbin_reject_enable = True
    match_net.dustbin_reject_margin = 0.0
    match_net.to(device).eval()
    logger.info("Device: %s", device)

    all_labels = []
    all_probs = []
    all_raw_k = []  
    all_k_scores = []
    iteration = 0
    with torch.no_grad():
        for batch in dataloader:
            iteration += 1
            
            # Send data to device and run inference
            batch = data_to_cuda(batch)
            outputs = match_net(batch)
            # Force dustbin handling to mirror classify_pairs behavior when checkpoints
            # don't set the flag reliably.
            outputs["has_dustbin"] = True
            has_dustbin = outputs.get("has_dustbin", False)
            # If there is a dustbin reduce number on points
            if has_dustbin:
                n1 = outputs["ns"][0] - 1
                n2 = outputs["ns"][1] - 1
            else:
                n1 = outputs["ns"][0]
                n2 = outputs["ns"][1]
            
            # Strip dustbin from matrices to mirror classify_pairs post-processing, which is important for consistent k_pred and k_score calculation.
            outputs["ds_mat"] = strip_dustbin_by_ns(outputs["ds_mat"], n1, n2)
            outputs["perm_mat"] = strip_dustbin_by_ns(outputs["perm_mat"], n1, n2)
            if "gt_perm_mat" in outputs:
                outputs["gt_perm_mat"] = strip_dustbin_by_ns(outputs["gt_perm_mat"], n1, n2)
            outputs["ns"] = [n1, n2]
            # get k_pred and k_score for this batch
            perm_mat = outputs["perm_mat"].detach()
            k_pred = perm_mat.sum(dim=(1, 2)).float()
            ns = outputs["ns"]
            min_points = torch.min(ns[0], ns[1]).float().clamp(min=1.0)
            k_score = (k_pred / min_points).clamp(0, 1)

            # Use learned authentication probability if available.
            if "auth_prob" in outputs:
                prob = outputs["auth_prob"].detach().view(-1).clamp(0, 1)
            else:
                prob = k_score
            all_probs.append(prob.cpu())
            all_labels.append(batch["label"].cpu())
            all_raw_k.append(k_pred.cpu()) 
            all_k_scores.append(k_score.cpu())
            if iteration % 5 == 0:
                logger.info("Processed %d batches...", iteration)
                

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_k_scores = torch.cat(all_k_scores).numpy()

    # Debug: Check label distribution
    logger.info("Total samples: %d", len(all_labels))
    logger.info("Genuine matches (label=1): %d", int(np.sum(all_labels == 1)))
    logger.info("Imposter matches (label=0): %d", int(np.sum(all_labels == 0)))
    logger.info("Unique labels: %s", np.unique(all_labels))
    
    # If no genuine matches, let's check the first few batches manually
    if np.sum(all_labels == 1) == 0:
        logger.warning("No genuine matches found! Checking first few batches...")
        debug_dataloader = get_dataloader(dataset, batch_size=8, shuffle=False, fix_seed=True)
        for i, batch in enumerate(debug_dataloader):
            if i >= 5:  # Check first 5 batches
                break
            batch = data_to_cuda(batch)
            labels = batch["label"].cpu().numpy()
            logger.info("Batch %d: labels=%s", i, labels)
                

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

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize one genuine match (label == 1) from the network
    for i, (label, prob) in enumerate(zip(all_labels, all_probs)):
        if label == 1:
            # Re-run the dataloader to get the corresponding batch and visualize
            count = 0
            for batch in get_dataloader(dataset, batch_size=8, shuffle=False, fix_seed=True):
                batch = data_to_cuda(batch)
                batch_label = batch["label"].cpu().numpy()[0]
                if batch_label == 1:
                    # Run the model to get outputs
                    with torch.no_grad():
                        outputs = match_net(batch)
                    # Mirror debug_final_match post-processing for dustbin handling.
                    has_dustbin = outputs.get("has_dustbin", False)
                    n1 = outputs["ns"][0][0] if isinstance(outputs["ns"][0], torch.Tensor) else outputs["ns"][0]
                    n2 = outputs["ns"][1][0] if isinstance(outputs["ns"][1], torch.Tensor) else outputs["ns"][1]
                    if has_dustbin:
                        n1 = n1 - 1
                        n2 = n2 - 1
                    outputs["ds_mat"] = strip_dustbin_by_ns(outputs["ds_mat"], n1, n2)
                    outputs["perm_mat"] = strip_dustbin_by_ns(outputs["perm_mat"], n1, n2)
                    if "gt_perm_mat" in outputs:
                        outputs["gt_perm_mat"] = strip_dustbin_by_ns(outputs["gt_perm_mat"], n1, n2)
                    outputs["ns"] = [n1, n2]

                    
                    # Get keypoints
                    if 'Ps' in batch:
                        kp0 = batch['Ps'][0][0].cpu().numpy()
                        kp1 = batch['Ps'][1][0].cpu().numpy()
                        n1_int = int(n1.item()) if isinstance(n1, torch.Tensor) else int(n1)
                        n2_int = int(n2.item()) if isinstance(n2, torch.Tensor) else int(n2)
                        kp0 = kp0[:n1_int]
                        kp1 = kp1[:n2_int]
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

                    
                    print(f"Genuine match visualization saved with {len(matches)} matches", flush=True)
                    print(f"Probability: {prob:.4f}", flush=True)
                    break
                count += 1
            break
    
    # Do the same for an imposter match (label == 0)
    for i, (label, prob) in enumerate(zip(all_labels, all_probs)):
        if label == 0:
            count = 0
            for batch in get_dataloader(dataset, batch_size=8, shuffle=True, fix_seed=True):
                batch = data_to_cuda(batch)
                batch_label = batch["label"].cpu().numpy()[0]
                if batch_label == 0:
                    with torch.no_grad():
                        outputs = match_net(batch)
                    # Mirror debug_final_match post-processing for dustbin handling.
                    has_dustbin = outputs.get("has_dustbin", False)
                    n1 = outputs["ns"][0][0] if isinstance(outputs["ns"][0], torch.Tensor) else outputs["ns"][0]
                    n2 = outputs["ns"][1][0] if isinstance(outputs["ns"][1], torch.Tensor) else outputs["ns"][1]
                    if has_dustbin:
                        n1 = n1 - 1
                        n2 = n2 - 1
                    outputs["ds_mat"] = strip_dustbin_by_ns(outputs["ds_mat"], n1, n2)
                    outputs["perm_mat"] = strip_dustbin_by_ns(outputs["perm_mat"], n1, n2)
                    if "gt_perm_mat" in outputs:
                        outputs["gt_perm_mat"] = strip_dustbin_by_ns(outputs["gt_perm_mat"], n1, n2)
                    outputs["ns"] = [n1, n2]
                    print(n1, n2)
                    
                    if 'Ps' in batch:
                        kp0 = batch['Ps'][0][0].cpu().numpy()
                        kp1 = batch['Ps'][1][0].cpu().numpy()
                        n1_int = int(n1.item()) if isinstance(n1, torch.Tensor) else int(n1)
                        n2_int = int(n2.item()) if isinstance(n2, torch.Tensor) else int(n2)
                        kp0 = kp0[:n1_int]
                        kp1 = kp1[:n2_int]
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
                    
           
                    
                    print(f"Imposter match visualization saved with {len(matches)} matches", flush=True)
                    print(f"Probability: {prob:.4f}", flush=True)
                    break
                count += 1
            break

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

    # Concatenate raw k values
    all_raw_k = torch.cat(all_raw_k).numpy()

    # Histogram of normalized k values
    plt.figure(figsize=(10, 6))
    genuine_probs = all_probs[all_labels == 1]
    imposter_probs = all_probs[all_labels == 0]

    # Use bins that cover the range from 0 to 1
    bins = np.linspace(0, 1, 30)

    plt.hist(imposter_probs, bins=bins, alpha=0.5, label='Imposter Matches', color='red')
    plt.hist(genuine_probs, bins=bins, alpha=0.5, label='Genuine Matches', color='green')

    plt.xlabel('Normalized k Value (k_pred/min_points)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Normalized k Values')
    plt.grid(alpha=0.3)
    plt.axvline(x=eer_threshold, color='black', linestyle='--', 
               label=f'EER Threshold ({eer_threshold:.3f})')
    plt.legend()
    plt.savefig(out_dir / "normalized_k_histogram.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Histogram of raw k values
    plt.figure(figsize=(10, 6))
    genuine_raw_k = all_raw_k[all_labels == 1]
    imposter_raw_k = all_raw_k[all_labels == 0]

    # Find appropriate bin range for raw k values
    max_k = np.max(all_raw_k) * 1.05  # Add a small margin
    bins = np.linspace(0, max_k, 30)

    plt.hist(imposter_raw_k, bins=bins, alpha=0.5, label='Imposter Matches', color='red')
    plt.hist(genuine_raw_k, bins=bins, alpha=0.5, label='Genuine Matches', color='green')

    plt.xlabel('Raw k Value (Number of Matched Keypoints)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Raw k Values')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "raw_k_histogram.png", bbox_inches="tight", pad_inches=0)
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

    print("Evaluation metrics:", flush=True)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}", flush=True)
        logger.info("%s: %.4f", k, v)

    # K-only rule: k == 0 => imposter, else genuine
    k_rule_preds = (all_raw_k > 0).astype(np.int32)
    if np.unique(all_labels).size >= 2:
        k_accuracy = accuracy_score(all_labels, k_rule_preds)
        k_precision = precision_score(all_labels, k_rule_preds)
        k_recall = recall_score(all_labels, k_rule_preds)
        k_f1 = f1_score(all_labels, k_rule_preds)
        k_tn, k_fp, k_fn, k_tp = confusion_matrix(all_labels, k_rule_preds).ravel()
        k_far = k_fp / (k_fp + k_tn) if (k_fp + k_tn) > 0 else 0.0
        k_frr = k_fn / (k_tp + k_fn) if (k_tp + k_fn) > 0 else 0.0

        k_metrics = {
            "k_rule_accuracy": k_accuracy,
            "k_rule_precision": k_precision,
            "k_rule_recall": k_recall,
            "k_rule_f1_score": k_f1,
            "k_rule_far": k_far,
            "k_rule_frr": k_frr,
        }

        print("K-rule metrics (k==0 => imposter):", flush=True)
        for k, v in k_metrics.items():
            print(f"{k}: {v:.4f}", flush=True)
            logger.info("%s: %.4f", k, v)
        k_rule_dir = out_dir / "k_rule"
        k_rule_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([k_metrics]).to_csv(k_rule_dir / "metrics.csv", index=False)

        # ROC/PR using normalized k scores
        k_fpr, k_tpr, _ = roc_curve(all_labels, all_k_scores)
        k_roc_auc = auc(k_fpr, k_tpr)
        k_prec_curve, k_rec_curve, _ = precision_recall_curve(all_labels, all_k_scores)
        k_pr_auc = auc(k_rec_curve, k_prec_curve)

        plt.figure()
        plt.plot(k_fpr, k_tpr, label=f"ROC AUC = {k_roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Normalized k)")
        plt.legend(loc="lower right")
        plt.savefig(k_rule_dir / "roc_curve.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        plt.figure()
        plt.plot(k_rec_curve, k_prec_curve, label=f"PR AUC = {k_pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Normalized k)")
        plt.legend(loc="lower left")
        plt.savefig(k_rule_dir / "pr_curve.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        # Histogram of normalized k scores
        plt.figure(figsize=(10, 6))
        k_genuine = all_k_scores[all_labels == 1]
        k_imposter = all_k_scores[all_labels == 0]
        bins = np.linspace(0, 1, 30)
        plt.hist(k_imposter, bins=bins, alpha=0.5, label='Imposter Matches', color='red')
        plt.hist(k_genuine, bins=bins, alpha=0.5, label='Genuine Matches', color='green')
        plt.xlabel('Normalized k Value (k_pred/min_points)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Normalized k Values (K-rule)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(k_rule_dir / "normalized_k_histogram.png", bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        logger.warning("K-rule metrics skipped: only one class present in labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained binary classifier")
    parser.add_argument(
        "--dataset",
        choices=["L3SFV2Augmented", "PolyU-DBII", "PolyU-DBI", "L3-SF"],
        default="L3SFV2Augmented",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory of the dataset. If omitted a sensible default is used.",
    )
    parser.add_argument(
        "--filter",
        choices=["none", "intersection", "inclusion"],
        default="none",
        help="Keypoint filter strategy. Use 'none' to keep all keypoints.",
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

    filter_value = args.filter.lower()
    if filter_value == "none":
        filter_value = None

    evaluate(args.dataset, data_root, filter=filter_value)
