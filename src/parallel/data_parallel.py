import torch
import torch.nn as nn
from .scatter_gather import scatter_kwargs, gather


class DataParallel(nn.DataParallel):
    """
    DataParallel wrapper with customized scatter/gather functions
    """
    def __init__(self, *args, **kwargs):
        super(DataParallel, self).__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

                            
class SeqDataParallel(DataParallel):
    """
    DataParallel that understands time-major padded sequences.

    Assumes the *first* positional arg is [T, B, …] and the second is
    `lengths` [B].  Those two are scattered differently; everything else
    falls back to the default behaviour.
    """

    def scatter(self, inputs, kwargs, device_ids):
        if len(inputs) >= 2:
            seq, lengths, *rest = inputs

            # only activate the special logic when shapes look right
            if seq.dim() >= 2 and lengths.dim() == 1 and seq.size(1) == lengths.size(0):
                batch_size = lengths.size(0)
                idx_chunks = torch.tensor_split(torch.arange(batch_size),
                                                len(device_ids))

                seq_chunks    = [seq[:, idx, ...].contiguous() for idx in idx_chunks]
                len_chunks    = [lengths[idx].contiguous()     for idx in idx_chunks]
                per_gpu_inps  = [(s, l, *rest) for s, l in zip(seq_chunks, len_chunks)]

                return scatter_kwargs(per_gpu_inps, kwargs, device_ids, dim=0)

        # fallback to vanilla behaviour
        return super().scatter(inputs, kwargs, device_ids)