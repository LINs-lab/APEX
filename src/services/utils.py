import torch
import typing
from typing import Optional, Union

from collections import OrderedDict, defaultdict

from .tools import create_logger

logger = create_logger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            ema_name = (
                model_name.replace("module.", "")
                if model_name.startswith("module.")
                else f"module.{model_name}"
            )
            if ema_name in ema_params:
                ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                raise KeyError(f"Parameter name {model_name} not found in EMA model!")


@torch.no_grad()
def update_ema_lora(
    model,
    adapter_name: str = "default",
    ema_adapter_name: str = "old",
    decay: float = 0.9999,
):
    """
    Update EMA LoRA adapter within the same model.
    """
    all_params = dict(model.named_parameters())

    ema_tensors = []
    model_tensors = []

    for name, param in all_params.items():
        if adapter_name in name and param.requires_grad:
            ema_name = name.replace(adapter_name, ema_adapter_name)
            if ema_name not in all_params:
                raise KeyError(f"EMA parameter {ema_name} not found for {name}")

            ema_param = all_params[ema_name]
            ema_tensors.append(ema_param.data)
            model_tensors.append(param.data.to(dtype=ema_param.dtype))

    if ema_tensors:
        torch._foreach_mul_(ema_tensors, decay)
        torch._foreach_add_(ema_tensors, model_tensors, alpha=1 - decay)
