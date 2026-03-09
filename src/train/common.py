from __future__ import annotations

import torch

from src.evaluation_metric import matching_classification_metrics, matching_metrics_from_counts


def loss_as_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(float(value), device=device)


def compose_total_loss(
    primary_loss,
    ks_loss,
    dustbin_loss,
    stage=None,
    stage1_ss_base_aux_loss=None,
    stage1_ss_base_aux_weight: float = 0.0,
):
    device = primary_loss.device

    if stage == 1:
        aux_loss = 0.0 if stage1_ss_base_aux_loss is None else stage1_ss_base_aux_loss
        return (
            loss_as_tensor(primary_loss, device)
            + (float(stage1_ss_base_aux_weight) * loss_as_tensor(aux_loss, device))
            + loss_as_tensor(ks_loss, device)
            + loss_as_tensor(dustbin_loss, device)
        )
    elif stage == 4:
        return (
            loss_as_tensor(primary_loss, device)
            + loss_as_tensor(ks_loss, device)
            + loss_as_tensor(dustbin_loss, device)
        )
    elif stage == 2:
        return loss_as_tensor(ks_loss, device)
    elif stage == 3:    
        return loss_as_tensor(dustbin_loss, device)
    return (
        loss_as_tensor(primary_loss, device)
        + loss_as_tensor(ks_loss, device)
        + loss_as_tensor(dustbin_loss, device)
    )


def batch_match_counts(outputs) -> dict[str, float]:
    batch_metrics = matching_classification_metrics(
        outputs["perm_mat"],
        outputs["gt_perm_mat"],
        outputs["ns"],
    )
    return {
        "tp": float(batch_metrics["tp"].sum().item()),
        "tn": float(batch_metrics["tn"].sum().item()),
        "fp": float(batch_metrics["fp"].sum().item()),
        "fn": float(batch_metrics["fn"].sum().item()),
    }


def counts_summary(tp: float, tn: float, fp: float, fn: float, device) -> dict:
    return matching_metrics_from_counts(
        torch.tensor(tp, device=device),
        torch.tensor(tn, device=device),
        torch.tensor(fp, device=device),
        torch.tensor(fn, device=device),
    )


def summary_scalars(summary: dict) -> dict[str, float]:
    return {
        "accuracy": float(summary["accuracy"].item()),
        "precision": float(summary["precision"].item()),
        "recall": float(summary["recall"].item()),
        "f1": float(summary["f1"].item()),
        "micro_f1": float(summary["micro_f1"].item()),
        "macro_f1": float(summary["macro_f1"].item()),
        "class0_precision": float(summary["per_class_precision"][0].item()),
        "class1_precision": float(summary["per_class_precision"][1].item()),
        "class0_recall": float(summary["per_class_recall"][0].item()),
        "class1_recall": float(summary["per_class_recall"][1].item()),
        "class0_f1": float(summary["per_class_f1"][0].item()),
        "class1_f1": float(summary["per_class_f1"][1].item()),
    }


def k_debug_scalars(outputs: dict) -> dict[str, float]:
    """Summarize K-related debug signals from model outputs.

    Returns an empty dict if K-related tensors are unavailable.
    """

    def _as_flat_float_tensor(value):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.detach().to(dtype=torch.float32).reshape(-1)
        if value.numel() == 0:
            return value
        finite = torch.isfinite(value)
        if not torch.all(finite):
            value = value[finite]
        return value

    gt_k = _as_flat_float_tensor(outputs.get("gt_ks"))
    pred_k = _as_flat_float_tensor(outputs.get("k_pred_count"))
    selected_k = _as_flat_float_tensor(outputs.get("k_match_count"))
    dustbin_pred_k = _as_flat_float_tensor(outputs.get("dustbin_k_pred_count"))

    ref = next(
        (x for x in (gt_k, pred_k, selected_k, dustbin_pred_k) if x is not None and x.numel() > 0),
        None,
    )
    if ref is None:
        return {}

    stats = {"sample_count": float(ref.numel())}

    if gt_k is not None and gt_k.numel() > 0:
        gt_k = gt_k.clamp(min=0.0)
        stats["gt_k_mean"] = float(gt_k.mean().item())
        stats["gt_zero_k_rate"] = float((gt_k <= 0.0).to(torch.float32).mean().item())

    if pred_k is not None and pred_k.numel() > 0:
        pred_k = pred_k.clamp(min=0.0)
        pred_k_round = torch.round(pred_k)
        stats["pred_k_mean"] = float(pred_k.mean().item())
        stats["pred_k_round_mean"] = float(pred_k_round.mean().item())
        stats["pred_zero_k_rate"] = float((pred_k_round <= 0.0).to(torch.float32).mean().item())

    if selected_k is not None and selected_k.numel() > 0:
        selected_k = selected_k.clamp(min=0.0)
        stats["selected_k_mean"] = float(selected_k.mean().item())
        stats["selected_zero_k_rate"] = float((selected_k <= 0.0).to(torch.float32).mean().item())

    if gt_k is not None and pred_k is not None and gt_k.numel() > 0 and pred_k.numel() > 0:
        n = min(gt_k.numel(), pred_k.numel())
        stats["pred_k_mae"] = float(torch.mean(torch.abs(pred_k[:n] - gt_k[:n])).item())

    if gt_k is not None and selected_k is not None and gt_k.numel() > 0 and selected_k.numel() > 0:
        n = min(gt_k.numel(), selected_k.numel())
        stats["selected_k_mae"] = float(torch.mean(torch.abs(selected_k[:n] - gt_k[:n])).item())

    if dustbin_pred_k is not None and dustbin_pred_k.numel() > 0:
        dustbin_pred_k = dustbin_pred_k.clamp(min=0.0)
        dustbin_pred_k_round = torch.round(dustbin_pred_k)
        stats["dustbin_k_mean"] = float(dustbin_pred_k.mean().item())
        stats["dustbin_k_round_mean"] = float(dustbin_pred_k_round.mean().item())
        stats["dustbin_k_zero_rate"] = float((dustbin_pred_k_round <= 0.0).to(torch.float32).mean().item())

    if gt_k is not None and dustbin_pred_k is not None and gt_k.numel() > 0 and dustbin_pred_k.numel() > 0:
        n = min(gt_k.numel(), dustbin_pred_k.numel())
        stats["dustbin_k_pred_mae"] = float(torch.mean(torch.abs(dustbin_pred_k[:n] - gt_k[:n])).item())

    for output_key, stat_key in (
        ("dustbin_k_mse_loss", "dustbin_k_mse_loss"),
        ("dustbin_k_mae", "dustbin_k_mae_loss"),
        ("dustbin_bce_loss", "dustbin_bce_loss"),
        ("dustbin_k_balance_err", "dustbin_k_balance_err"),
    ):
        scalar = _as_flat_float_tensor(outputs.get(output_key))
        if scalar is not None and scalar.numel() > 0:
            stats[stat_key] = float(scalar.mean().item())

    return stats
