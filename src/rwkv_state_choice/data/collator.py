"""Tokenize variable-choice examples into padded tensors."""

from __future__ import annotations

from typing import Any, TypedDict

import torch

from rwkv_state_choice.schema import ChoiceExample


class ChoiceBatch(TypedDict):
    problem_input_ids: torch.Tensor
    problem_attention_mask: torch.Tensor
    candidate_input_ids: torch.Tensor
    candidate_attention_mask: torch.Tensor
    candidate_mask: torch.Tensor
    targets: torch.Tensor
    example_ids: list[str]
    metadata: list[dict[str, Any]]


class ChoiceCollator:
    """Create right-padded question and candidate token tensors.

    Candidate tensors have shape `[batch, max_candidates, sequence_length]`.
    `candidate_mask` marks real candidates when examples have different choice counts.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_problem_tokens: int = 256,
        max_candidate_tokens: int = 128,
    ) -> None:
        if max_problem_tokens <= 0 or max_candidate_tokens <= 0:
            raise ValueError("token limits must be positive")
        if tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer must define a pad_token_id")

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_problem_tokens = max_problem_tokens
        self.max_candidate_tokens = max_candidate_tokens
        self.pad_token_id = int(tokenizer.pad_token_id)

    def _tokenize(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def __call__(self, examples: list[ChoiceExample]) -> ChoiceBatch:
        if not examples:
            raise ValueError("cannot collate an empty batch")

        problem_tokens = self._tokenize(
            [example.problem for example in examples], self.max_problem_tokens
        )

        batch_size = len(examples)
        candidate_counts = [len(example.candidates) for example in examples]
        max_candidates = max(candidate_counts)
        flat_candidates = [candidate for example in examples for candidate in example.candidates]
        flat_tokens = self._tokenize(flat_candidates, self.max_candidate_tokens)

        candidate_length = flat_tokens["input_ids"].shape[1]
        candidate_input_ids = torch.full(
            (batch_size, max_candidates, candidate_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        candidate_attention_mask = torch.zeros_like(candidate_input_ids)
        candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)

        offset = 0
        for batch_index, count in enumerate(candidate_counts):
            end = offset + count
            candidate_input_ids[batch_index, :count] = flat_tokens["input_ids"][offset:end]
            candidate_attention_mask[batch_index, :count] = flat_tokens["attention_mask"][offset:end]
            candidate_mask[batch_index, :count] = True
            offset = end

        targets = torch.tensor(
            [-100 if example.target is None else example.target for example in examples],
            dtype=torch.long,
        )

        return {
            "problem_input_ids": problem_tokens["input_ids"],
            "problem_attention_mask": problem_tokens["attention_mask"],
            "candidate_input_ids": candidate_input_ids,
            "candidate_attention_mask": candidate_attention_mask,
            "candidate_mask": candidate_mask,
            "targets": targets,
            "example_ids": [example.example_id for example in examples],
            "metadata": [dict(example.metadata) for example in examples],
        }

