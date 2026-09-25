"""Metrics for candidate-choice evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch

from rwkv_state_choice.data.collator import ChoiceBatch
from rwkv_state_choice.model.matcher import StateChoiceMatcher


def move_batch(batch: ChoiceBatch, device: torch.device) -> ChoiceBatch:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_choice_model(
    model: StateChoiceMatcher,
    batches: Iterable[ChoiceBatch],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    by_subject: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_candidate_count: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    for cpu_batch in batches:
        batch = move_batch(cpu_batch, device)
        targets = batch["targets"]
        labelled = targets != -100
        if not bool(labelled.any()):
            continue

        scores = model(batch)["scores"]
        predictions = scores.argmax(dim=-1)
        matches = predictions.eq(targets)
        counts = batch["candidate_mask"].sum(dim=-1)

        for index in torch.nonzero(labelled, as_tuple=False).flatten().tolist():
            is_correct = int(matches[index].item())
            subject = str(batch["metadata"][index].get("subject", "unknown"))
            count = int(counts[index].item())
            total += 1
            correct += is_correct
            by_subject[subject][0] += is_correct
            by_subject[subject][1] += 1
            by_candidate_count[count][0] += is_correct
            by_candidate_count[count][1] += 1

    if total == 0:
        raise ValueError("evaluation produced no labelled examples")

    return {
        "accuracy": correct / total,
        "num_examples": total,
        "accuracy_by_subject": {
            subject: values[0] / values[1]
            for subject, values in sorted(by_subject.items())
        },
        "accuracy_by_candidate_count": {
            str(count): values[0] / values[1]
            for count, values in sorted(by_candidate_count.items())
        },
    }

