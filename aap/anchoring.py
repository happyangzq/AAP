"""Core asymmetric anchoring operations used by AAP."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_error(
    projected_features: torch.Tensor,
    anchor_features: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return patch-wise cosine distance, ``1 - cosine_similarity``."""
    projected = F.normalize(projected_features.float(), dim=-1, eps=eps)
    anchor = F.normalize(anchor_features.detach().float(), dim=-1, eps=eps)
    return 1.0 - (projected * anchor).sum(dim=-1)


def asymmetric_anchor_objective(
    projected_features: torch.Tensor,
    anchor_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    real_label: int = 0,
    tampered_label: int = 2,
) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    """Compute ARL for real images and error maps for tampered images.

    Args:
        projected_features: Tensor shaped ``[B, L, N, D]``.
        anchor_features: Frozen truth-anchor tensor shaped ``[B, N, D]``.
        labels: Image-level class labels shaped ``[B]``.

    Returns:
        The scalar asymmetric anchoring loss and one optional patch error map
        per sample. Only tampered samples receive an error map.
    """
    if projected_features.ndim != 4:
        raise ValueError("projected_features must have shape [B, L, N, D]")
    if anchor_features.ndim != 3:
        raise ValueError("anchor_features must have shape [B, N, D]")
    if projected_features.shape[0] != anchor_features.shape[0]:
        raise ValueError("projected and anchor batch sizes must match")
    if projected_features.shape[2:] != anchor_features.shape[1:]:
        raise ValueError("projected and anchor patch features must match")

    errors = cosine_error(projected_features, anchor_features[:, None])
    real_mask = labels == real_label

    if real_mask.any():
        loss = errors[real_mask].mean()
    else:
        # Keep the alignment projector in the graph for homogeneous batches.
        loss = projected_features.sum() * 0.0

    error_maps: list[torch.Tensor | None] = []
    for index, label in enumerate(labels):
        if int(label.item()) == tampered_label:
            error_maps.append(errors[index, -1].detach())
        else:
            error_maps.append(None)

    return loss, error_maps
