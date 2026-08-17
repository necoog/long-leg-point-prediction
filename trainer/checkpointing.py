from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    # Common checkpoint formats:
    # - plain state_dict
    # - {"state_dict": ...}
    # - {"model_state_dict": ...}
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]

    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Heuristic: looks like a state_dict already
        return ckpt  # type: ignore[return-value]

    raise ValueError(
        "Unrecognized checkpoint format. Expected a state_dict or dict with "
        "`model_state_dict` / `state_dict`."
    )


def maybe_fix_module_prefix(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Handles common DataParallel prefix mismatch ("module.").
    """
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
    model_has_module = any(k.startswith("module.") for k in model.state_dict().keys())

    if ckpt_has_module and not model_has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    if (not ckpt_has_module) and model_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_weights_into_model(
    *,
    model: nn.Module,
    checkpoint_path: str,
    strict: bool,
) -> Tuple[Any, list[str], list[str]]:
    """
    Returns: (raw_ckpt, missing_keys, unexpected_keys)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = maybe_fix_module_prefix(extract_state_dict(ckpt), model)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return ckpt, missing, unexpected

