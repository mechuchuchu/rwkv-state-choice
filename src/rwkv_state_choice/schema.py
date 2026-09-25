"""Benchmark-independent examples for choice and answer matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ChoiceExample:
    """One problem with a variable number of candidate answers.

    `target` is the zero-based candidate index, or `None` for inference-only data.
    Benchmark-specific fields belong in `metadata`; they are never model inputs.
    """

    problem: str
    candidates: Sequence[str]
    target: int | None = None
    example_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("problem must be a non-empty string")

        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("each example must have at least one candidate")
        if any(not isinstance(candidate, str) or not candidate.strip() for candidate in candidates):
            raise ValueError("every candidate must be a non-empty string")
        object.__setattr__(self, "candidates", candidates)

        if self.target is not None:
            if isinstance(self.target, bool) or not isinstance(self.target, Integral):
                raise TypeError("target must be an integer candidate index or None")
            target = int(self.target)
            if not 0 <= target < len(candidates):
                raise ValueError(
                    f"target index {target} is outside {len(candidates)} candidates"
                )
            object.__setattr__(self, "target", target)

        if not isinstance(self.example_id, str):
            raise TypeError("example_id must be a string")
        object.__setattr__(self, "metadata", dict(self.metadata))

