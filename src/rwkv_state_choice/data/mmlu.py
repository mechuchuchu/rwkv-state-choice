"""Adapter for the Hugging Face `cais/mmlu` dataset."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

from rwkv_state_choice.schema import ChoiceExample

MMLU_DATASET_ID = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"

_STANDARD_SPLITS = {"dev", "validation", "test"}
_SUPPORTED_SPLITS = _STANDARD_SPLITS | {"auxiliary_train"}
def _answer_index(answer: Any) -> int:
    if isinstance(answer, Integral) and not isinstance(answer, bool):
        return int(answer)
    if isinstance(answer, str):
        label = answer.strip().upper()
        if len(label) == 1 and "A" <= label <= "Z":
            return ord(label) - ord("A")
    raise ValueError(f"unrecognized MMLU answer value: {answer!r}")


def _to_example(
    row: Mapping[str, Any],
    index: int,
    split: str,
    dataset_id: str,
    revision: str,
) -> ChoiceExample:
    try:
        problem = row["question"]
        candidates = row["choices"]
        answer = row["answer"]
        subject = row["subject"]
    except KeyError as error:
        raise ValueError(f"MMLU row is missing required column {error.args[0]!r}") from error

    if not isinstance(candidates, (list, tuple)):
        raise TypeError("MMLU choices must be a sequence of strings")
    candidates = tuple(candidates)
    target = _answer_index(answer)
    return ChoiceExample(
        problem=problem,
        candidates=candidates,
        target=target,
        example_id=f"mmlu:{split}:{index}",
        metadata={
            "benchmark": "mmlu",
            "split": split,
            "subject": str(subject),
            "source_dataset": dataset_id,
            "source_revision": revision,
            "is_auxiliary_train": split == "auxiliary_train",
        },
    )


def load_mmlu_split(
    split: str,
    *,
    dataset_id: str = MMLU_DATASET_ID,
    revision: str = MMLU_REVISION,
    cache_dir: str | None = None,
    max_examples: int | None = None,
) -> list[ChoiceExample]:
    """Load and normalize one MMLU split.

    `auxiliary_train` is a separate Hub config whose remote split is named `train`.
    The standard `all` config provides only `dev`, `validation`, and `test`.
    """
    if split not in _SUPPORTED_SPLITS:
        supported = ", ".join(sorted(_SUPPORTED_SPLITS))
        raise ValueError(f"unsupported MMLU split {split!r}; choose one of: {supported}")
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative or None")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Loading MMLU requires the `datasets` package; install the project dependencies first"
        ) from error

    if split == "auxiliary_train":
        config_name, hub_split = "auxiliary_train", "train"
    else:
        config_name, hub_split = "all", split

    dataset = load_dataset(
        dataset_id,
        name=config_name,
        split=hub_split,
        revision=revision,
        cache_dir=cache_dir,
    )
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    return [
        _to_example(row, index, split, dataset_id, revision)
        for index, row in enumerate(dataset)
    ]
