from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

DUSTBIN_ROW = -1
DUSTBIN_COL = -1


def add_dustbin(scores: Tensor, fill: float = 0.0) -> Tensor:
    *batch, n1, n2 = scores.shape
    out = scores.new_full((*batch, n1 + 1, n2 + 1), fill)
    out[..., :n1, :n2] = scores
    return out


def strip_dustbin(scores: Tensor) -> Tensor:
    return scores[..., :-1, :-1]


def dustbin_row(scores: Tensor) -> Tensor:
    return scores[..., DUSTBIN_ROW, :]


def dustbin_col(scores: Tensor) -> Tensor:
    return scores[..., :, DUSTBIN_COL]


def _ns_to_list(ns, batch_size):
    if isinstance(ns, torch.Tensor):
        if ns.dim() == 0:
            return [int(ns.item())] * batch_size
        return [int(x.item()) for x in ns.view(-1)]
    if isinstance(ns, (list, tuple)):
        if len(ns) == 0:
            return [0] * batch_size
        return [int(x.item()) if hasattr(x, "item") else int(x) for x in ns]
    try:
        return [int(ns)] * batch_size
    except (TypeError, ValueError):
        return [0] * batch_size


def strip_dustbin_by_ns(scores: Tensor, n1, n2) -> Tensor:
    """Strip dustbin row/col using true per-sample sizes.

    Supports 2D or 3D tensors/arrays. For 3D input, n1/n2 can be
    tensors, lists, or scalars; output is padded to max(n1)/max(n2).
    """
    if scores is None:
        return None

    is_torch = isinstance(scores, torch.Tensor)
    if is_torch:
        device = scores.device
        dtype = scores.dtype

    if scores.ndim == 2:
        n1_i = _ns_to_list(n1, 1)[0]
        n2_i = _ns_to_list(n2, 1)[0]
        if scores.shape[-2] >= n1_i + 1 and scores.shape[-1] >= n2_i + 1:
            cropped = scores[:n1_i + 1, :n2_i + 1]
            return cropped[:-1, :-1]
        return scores[:n1_i, :n2_i]

    if scores.ndim != 3:
        return scores

    batch_size = scores.shape[0]
    n1_list = _ns_to_list(n1, batch_size)
    n2_list = _ns_to_list(n2, batch_size)
    max_n1 = max(n1_list) if batch_size > 0 else 0
    max_n2 = max(n2_list) if batch_size > 0 else 0

    if is_torch:
        padded_blocks = []
        for b in range(batch_size):
            nb1 = int(n1_list[b])
            nb2 = int(n2_list[b])
            block = scores[b]
            if block.shape[-2] >= nb1 + 1 and block.shape[-1] >= nb2 + 1:
                block = block[:nb1 + 1, :nb2 + 1]
                block = block[:-1, :-1]
            else:
                block = block[:nb1, :nb2]
            pad_h = max_n1 - nb1
            pad_w = max_n2 - nb2
            if pad_h > 0 or pad_w > 0:
                block = F.pad(block, (0, pad_w, 0, pad_h))
            padded_blocks.append(block)
        return torch.stack(padded_blocks, dim=0) if padded_blocks else torch.zeros((0, max_n1, max_n2), device=device, dtype=dtype)

    out = np.zeros((batch_size, max_n1, max_n2), dtype=scores.dtype)
    for b in range(batch_size):
        nb1 = int(n1_list[b])
        nb2 = int(n2_list[b])
        block = scores[b]
        if block.shape[-2] >= nb1 + 1 and block.shape[-1] >= nb2 + 1:
            block = block[:nb1 + 1, :nb2 + 1]
            block = block[:-1, :-1]
        else:
            block = block[:nb1, :nb2]
        out[b, :nb1, :nb2] = block
    return out
