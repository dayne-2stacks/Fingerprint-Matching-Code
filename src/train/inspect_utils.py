from __future__ import annotations

import numpy as np
import torch


def infer_batch_size(batch) -> int:
    if isinstance(batch, dict):
        if "batch_size" in batch:
            try:
                return int(batch["batch_size"])
            except (TypeError, ValueError):
                pass
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
            if isinstance(value, list) and value and isinstance(value[0], torch.Tensor) and value[0].dim() > 0:
                return int(value[0].shape[0])
    return 1


def slice_value(value, idx: int, batch_size: int):
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
        return {k: slice_value(v, idx, batch_size) for k, v in value.items()}
    return value


def slice_batch(batch, idx: int):
    batch_size = infer_batch_size(batch)
    if isinstance(batch, dict):
        return {k: slice_value(v, idx, batch_size) for k, v in batch.items()}
    return batch


def extract_label(sample):
    if not isinstance(sample, dict) or "label" not in sample:
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


def find_pair_index(batch, pair_type: str):
    if pair_type == "any":
        return 0
    if not isinstance(batch, dict) or "label" not in batch:
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
