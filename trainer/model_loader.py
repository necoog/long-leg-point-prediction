from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict

import torch.nn as nn


def load_model_from_file(
    model_def_path: str | os.PathLike[str],
    *,
    num_outputs: int,
    in_chans: int,
    pretrained: bool,
    extra_kwargs: Dict[str, Any] | None = None,
) -> nn.Module:
    """
    Loads a model-definition python file from disk.

    Contract:
      - file must expose `create_model(num_outputs=..., in_chans=..., pretrained=..., **kwargs)`
      - create_model must return torch.nn.Module
    """
    model_def_path = str(model_def_path)
    spec = importlib.util.spec_from_file_location("model_def", model_def_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model definition from: {model_def_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "create_model"):
        raise AttributeError(
            f"Model definition file must expose `create_model(...)`: {model_def_path}"
        )

    kwargs = dict(extra_kwargs or {})
    model = module.create_model(  # type: ignore[attr-defined]
        num_outputs=num_outputs,
        in_chans=in_chans,
        pretrained=pretrained,
        **kwargs,
    )

    if not isinstance(model, nn.Module):
        raise TypeError("create_model(...) must return a torch.nn.Module")

    return model

