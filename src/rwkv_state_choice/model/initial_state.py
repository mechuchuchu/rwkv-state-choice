"""Trainable shared initial recurrent state for all examples."""

from __future__ import annotations

import torch
from torch import nn


class GlobalInitialState(nn.Module):
    """Learn one zero-initialized RWKV cache shared across question examples.

    WKV stays FP32, matching the downloaded checkpoint configuration. Shift
    parameters are stored in FP32 and cast to the backbone activation dtype when a
    fresh cache is built for a batch.
    """

    def __init__(self, config) -> None:
        super().__init__()
        if config.wkv_state_dtype != "float32":
            raise ValueError("GlobalInitialState currently expects FP32 WKV cache state")

        self.num_layers = int(config.num_hidden_layers)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.head_dim)
        self.hidden_size = int(config.hidden_size)

        self.wkv = nn.Parameter(
            torch.zeros(self.num_layers, self.num_heads, self.head_dim, self.head_dim)
        )
        self.att_shift = nn.Parameter(torch.zeros(self.num_layers, self.hidden_size))
        self.ffn_shift = nn.Parameter(torch.zeros(self.num_layers, self.hidden_size))

    def build_cache(self, core: nn.Module, batch_size: int):
        """Make a new mutable RWKV cache seeded from the learned parameters."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        activation_dtype = core.emb.weight.dtype
        device = core.emb.weight.device
        cache = core.allocate_state(batch_size, device=device, dtype=activation_dtype)

        for layer_idx, layer in enumerate(cache.layers):
            states = layer.recurrent_states
            states[cache.WKV] = self.wkv[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            states[cache.ATT_SHIFT] = (
                self.att_shift[layer_idx]
                .to(dtype=activation_dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            states[cache.FFN_SHIFT] = (
                self.ffn_shift[layer_idx]
                .to(dtype=activation_dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        return cache

