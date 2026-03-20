from __future__ import annotations

import torch

from src.evaluation_metric import matching_classification_metrics, matching_metrics_from_counts


def compose_total_loss(
    primary_loss,
    ks_loss,
    dustbin_loss=None,
    stage=None,
):
    dustbin_loss = dustbin_loss if dustbin_loss is not None else 0.0
    if stage == 1 or stage == 4:
        return primary_loss + ks_loss + dustbin_loss
    elif stage == 2:
        return ks_loss
    elif stage == 3:
        return primary_loss + dustbin_loss



def batch_match_counts(outputs) -> dict[str, float]:
    batch_metrics = matching_classification_metrics(
        outputs["perm_mat"],
        outputs["gt_perm_mat"],
        outputs["ns"],
    )
    return {key: float(batch_metrics[key].sum().item()) 
            for key in ["tp", "tn", "fp", "fn"]}


def counts_summary(tp: float, tn: float, fp: float, fn: float, device) -> dict:
    return matching_metrics_from_counts(
        *[torch.tensor(val, device=device) for val in [tp, tn, fp, fn]]
    )


def summary_scalars(summary: dict) -> dict[str, float]:
    scalar_keys = ["accuracy", "precision", "recall", "f1", "micro_f1", "macro_f1"]
    result = {key: float(summary[key].item()) for key in scalar_keys}
    
    for class_idx in [0, 1]:
        result[f"class{class_idx}_precision"] = float(summary["per_class_precision"][class_idx].item())
        result[f"class{class_idx}_recall"] = float(summary["per_class_recall"][class_idx].item())
        result[f"class{class_idx}_f1"] = float(summary["per_class_f1"][class_idx].item())
    
    return result

