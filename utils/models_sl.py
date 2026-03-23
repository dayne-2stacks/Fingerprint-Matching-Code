import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def save_model(model, path):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    torch.save(model.state_dict(), path)


def load_model(model, path, strict=False):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
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

    # Filter out any per-parameter state whose tensor shapes don't match the current parameters.
    saved_groups = state.get("param_groups", [])
    current_groups = optimizer.param_groups
    if len(saved_groups) != len(current_groups):
        print(
            "Could not load optimizer state: loaded state dict has a different number of parameter groups. "
            "Starting with fresh optimizer."
        )
        return

    new_state = {"state": {}, "param_groups": saved_groups}
    for key, value in state.items():
        if key not in ("state", "param_groups"):
            new_state[key] = value

    for group_idx, (saved_group, current_group) in enumerate(zip(saved_groups, current_groups)):
        saved_params = saved_group.get("params", [])
        current_params = current_group.get("params", [])
        if len(saved_params) != len(current_params):
            print(
                "Could not load optimizer state: loaded state dict has different parameter counts in a group. "
                "Starting with fresh optimizer."
            )
            return

        for param_idx, (saved_param_id, current_param) in enumerate(zip(saved_params, current_params)):
            state_entry = state.get("state", {}).get(saved_param_id)
            if not state_entry:
                continue

            shape_ok = True
            for state_key, state_value in state_entry.items():
                if torch.is_tensor(state_value) and state_value.shape != current_param.shape:
                    shape_ok = False
                    break

            if shape_ok:
                new_state["state"][saved_param_id] = state_entry
            else:
                print(
                    "Skipping optimizer state for param group {} index {} due to shape mismatch.".format(
                        group_idx, param_idx
                    )
                )

    try:
        optimizer.load_state_dict(new_state)
    except ValueError as e:
        print(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
