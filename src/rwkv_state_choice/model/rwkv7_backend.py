"""Small adapter around the downloaded Transformers RWKV-7 implementation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class RWKV7Backbone(nn.Module):
    """Load the base checkpoint and expose its recurrent `Rwkv7Model` core."""

    def __init__(self, core: nn.Module, tokenizer) -> None:
        super().__init__()
        self.core = core
        self.tokenizer = tokenizer
        self.core.requires_grad_(False)
        self.core.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> "RWKV7Backbone":
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"RWKV-7 model directory does not exist: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        causal_lm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
        )
        if not hasattr(causal_lm, "rwkv7"):
            raise TypeError("loaded checkpoint does not expose the expected `rwkv7` core")

        core = causal_lm.rwkv7
        del causal_lm
        core.to(device)
        return cls(core, tokenizer)

    @property
    def hidden_size(self) -> int:
        return int(self.core.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return self.core.emb.weight.device

    @property
    def activation_dtype(self) -> torch.dtype:
        return self.core.emb.weight.dtype

    def train(self, mode: bool = True) -> "RWKV7Backbone":
        # The backbone stays in inference mode while autograd differentiates through
        # it into the trainable initial state and projection heads.
        self.training = mode
        self.core.eval()
        return self

