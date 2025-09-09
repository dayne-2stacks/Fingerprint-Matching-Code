import torch
from torch.nn import DataParallel


def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    torch.save(model.state_dict(), path)


def load_model(model, path, strict=False):
    if isinstance(model, DataParallel):
        module = model.module
    else:
        module = model

    state_dict = torch.load(path, map_location='cpu')
    model_dict = module.state_dict()
    # Filter out keys with mismatched shapes
    filtered_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            print(
                f"Skipping loading parameter: {k} due to shape mismatch ({v.shape} vs {model_dict.get(k, None).shape if k in model_dict else 'N/A'})"
            )

    missing_keys, unexpected_keys = module.load_state_dict(filtered_dict, strict=strict)
    if len(unexpected_keys) > 0:
        print(
            'Warning: Unexpected key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in unexpected_keys))
        )
    if len(missing_keys) > 0:
        print(
            'Warning: Missing key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in missing_keys))
        )


def load_optimizer(optimizer, path):
    """Load an optimizer state dict, skipping if param groups mismatch.

    This handles cases where the optimizer was saved for a model with
    different parameter groups (e.g. new layers added). If the file cannot be
    loaded or the group counts differ, the optimizer is left in its freshly
    initialized state.
    """
    try:
        state = torch.load(path, map_location="cpu")
    except FileNotFoundError as e:
        print(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
        return

    # If the number of parameter groups doesn't match, skip loading
    saved_groups = len(state.get("param_groups", []))
    current_groups = len(optimizer.param_groups)
    if saved_groups != current_groups:
        print(
            "Could not load optimizer state: loaded state dict has a different number of parameter groups. "
            "Starting with fresh optimizer."
        )
        return

    try:
        optimizer.load_state_dict(state)
    except ValueError as e:
        print(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
