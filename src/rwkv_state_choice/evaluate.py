"""Evaluate a saved state-choice adapter on MMLU validation or test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from rwkv_state_choice.checkpoint import load_adapter
from rwkv_state_choice.data import ChoiceCollator, load_mmlu_split
from rwkv_state_choice.evaluation import evaluate_choice_model
from rwkv_state_choice.model import RWKV7Backbone, StateChoiceMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--adapter", required=True, help="directory containing adapter.safetensors")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_config = config["model"]
    data_config = config["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))

    backbone = RWKV7Backbone.from_pretrained(
        model_config["path"], dtype=dtype, device=device
    )
    metadata_path = Path(args.adapter) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = StateChoiceMatcher(
        backbone,
        projection_dim=int(metadata["projection_dim"]),
        temperature=float(metadata["temperature"]),
    ).to(device)
    loaded_metadata = load_adapter(args.adapter, model)
    if loaded_metadata["base_model_revision"] != model_config["revision"]:
        raise ValueError("adapter was trained against a different base model revision")

    examples = load_mmlu_split(
        args.split,
        dataset_id=data_config["dataset_id"],
        revision=data_config["revision"],
    )
    collator = ChoiceCollator(
        backbone.tokenizer,
        max_problem_tokens=int(data_config["max_problem_tokens"]),
        max_candidate_tokens=int(data_config["max_candidate_tokens"]),
    )
    batches = DataLoader(
        examples,
        batch_size=int(config["training"].get("eval_batch_size", 1)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    metrics = evaluate_choice_model(model, batches, device)
    print(json.dumps({"split": args.split, **metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
