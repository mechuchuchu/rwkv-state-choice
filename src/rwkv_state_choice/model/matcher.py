"""Question-to-candidate compatibility model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rwkv_state_choice.data.collator import ChoiceBatch
from rwkv_state_choice.model.initial_state import GlobalInitialState
from rwkv_state_choice.model.rwkv7_backend import RWKV7Backbone


def _last_real_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("cannot pool a sequence with no non-padding tokens")
    positions = lengths - 1
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, positions]


class StateChoiceMatcher(nn.Module):
    """Score a variable number of candidates with normalized dot products."""

    def __init__(
        self,
        backbone: RWKV7Backbone,
        *,
        projection_dim: int = 512,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.backbone = backbone
        self.initial_state = GlobalInitialState(backbone.core.config)
        self.query_projection = nn.Linear(backbone.hidden_size, projection_dim, bias=False)
        self.candidate_projection = nn.Linear(backbone.hidden_size, projection_dim, bias=False)
        self.temperature = float(temperature)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.candidate_projection.weight)

    def train(self, mode: bool = True) -> "StateChoiceMatcher":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, batch: ChoiceBatch) -> dict[str, torch.Tensor]:
        problem_ids = batch["problem_input_ids"]
        problem_mask = batch["problem_attention_mask"]
        candidate_ids = batch["candidate_input_ids"]
        candidate_attention_mask = batch["candidate_attention_mask"]
        candidate_mask = batch["candidate_mask"].to(device=problem_ids.device, dtype=torch.bool)

        batch_size = problem_ids.shape[0]
        num_candidates = candidate_ids.shape[1]

        # This branch stays grad-enabled so loss can update the initial cache through
        # the frozen RWKV operations. Do not wrap it in `torch.no_grad()`.
        initial_cache = self.initial_state.build_cache(self.backbone.core, batch_size)
        query_output = self.backbone.core(
            input_ids=problem_ids,
            attention_mask=problem_mask,
            state=initial_cache,
            use_cache=True,
        )
        query_features = _last_real_token(query_output.last_hidden_state, problem_mask)
        query_embeddings = F.normalize(
            self.query_projection(query_features.float()), dim=-1
        )

        flat_ids = candidate_ids.reshape(batch_size * num_candidates, -1)
        flat_attention_mask = candidate_attention_mask.reshape(batch_size * num_candidates, -1)
        valid_candidates = candidate_mask.reshape(-1)
        valid_indices = torch.nonzero(valid_candidates, as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            raise ValueError("batch contains no candidate answers")

        # Candidate representations are fixed because the backbone is frozen and
        # candidates start from zero state. Avoid retaining their backbone graph.
        with torch.no_grad():
            candidate_output = self.backbone.core(
                input_ids=flat_ids.index_select(0, valid_indices),
                attention_mask=flat_attention_mask.index_select(0, valid_indices),
                use_cache=False,
            )
            candidate_features = _last_real_token(
                candidate_output.last_hidden_state,
                flat_attention_mask.index_select(0, valid_indices),
            )
            all_candidate_features = torch.zeros(
                batch_size * num_candidates,
                self.backbone.hidden_size,
                device=candidate_features.device,
                dtype=candidate_features.dtype,
            )
            all_candidate_features.index_copy_(0, valid_indices, candidate_features)

        candidate_embeddings = F.normalize(
            self.candidate_projection(
                all_candidate_features.view(batch_size, num_candidates, -1).float()
            ),
            dim=-1,
        )
        scores = torch.einsum("bd,bnd->bn", query_embeddings, candidate_embeddings)
        scores = scores / self.temperature
        scores = scores.masked_fill(~candidate_mask, -torch.inf)

        return {
            "scores": scores,
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings,
        }

