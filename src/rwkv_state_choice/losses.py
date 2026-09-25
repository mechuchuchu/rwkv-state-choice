"""Objectives for variable-choice matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def choice_cross_entropy(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over candidates; target `-100` marks inference-only rows."""
    labelled = targets != -100
    if not bool(labelled.any()):
        raise ValueError("batch has no labelled examples")
    return F.cross_entropy(scores[labelled], targets[labelled])

