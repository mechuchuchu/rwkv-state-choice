"""Save and restore only the trainable matching adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from rwkv_state_choice.model.matcher import StateChoiceMatcher


def save_adapter(
    directory: str | Path,
    model: StateChoiceMatcher,
    metadata: Mapping[str, Any],
) -> None:
    """Save state and projection weights without copying the frozen backbone."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    tensors = {
        "initial_state.wkv": model.initial_state.wkv.detach().cpu().contiguous(),
        "initial_state.att_shift": model.initial_state.att_shift.detach().cpu().contiguous(),
        "initial_state.ffn_shift": model.initial_state.ffn_shift.detach().cpu().contiguous(),
        "query_projection.weight": model.query_projection.weight.detach().cpu().contiguous(),
        "candidate_projection.weight": model.candidate_projection.weight.detach().cpu().contiguous(),
    }
    save_file(tensors, str(directory / "adapter.safetensors"))

    saved_metadata = {
        "schema_version": 1,
        "projection_dim": model.query_projection.out_features,
        "temperature": model.temperature,
        **dict(metadata),
    }
    (directory / "metadata.json").write_text(
        json.dumps(saved_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_adapter_metadata(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_adapter(directory: str | Path, model: StateChoiceMatcher) -> dict[str, Any]:
    """Copy adapter tensors into a model and return the recorded metadata."""
    directory = Path(directory)
    device = model.backbone.device
    tensors = load_file(str(directory / "adapter.safetensors"), device=str(device))
    expected = {
        "initial_state.wkv": model.initial_state.wkv,
        "initial_state.att_shift": model.initial_state.att_shift,
        "initial_state.ffn_shift": model.initial_state.ffn_shift,
        "query_projection.weight": model.query_projection.weight,
        "candidate_projection.weight": model.candidate_projection.weight,
    }
    if tensors.keys() != expected.keys():
        missing = sorted(expected.keys() - tensors.keys())
        unexpected = sorted(tensors.keys() - expected.keys())
        raise ValueError(f"adapter tensor names mismatch; missing={missing}, unexpected={unexpected}")

    with torch.no_grad():
        for name, parameter in expected.items():
            value = tensors[name]
            if value.shape != parameter.shape:
                raise ValueError(
                    f"adapter tensor {name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(dtype=parameter.dtype))

    return read_adapter_metadata(directory)

