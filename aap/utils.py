"""Shared constants and device helpers."""

import torch

from .llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


def dict_to_cuda(batch: dict) -> dict:
    """Move tensor values (including tensor lists) to the active CUDA device."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda(non_blocking=True)
        elif (
            isinstance(value, list)
            and value
            and isinstance(value[0], torch.Tensor)
        ):
            batch[key] = [element.cuda(non_blocking=True) for element in value]
    return batch
