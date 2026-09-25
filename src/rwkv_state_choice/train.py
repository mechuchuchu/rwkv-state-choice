"""Train the initial recurrent state and matching projection heads."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rwkv_state_choice.checkpoint import save_adapter
from rwkv_state_choice.data import ChoiceCollator, load_mmlu_split
from rwkv_state_choice.evaluation import evaluate_choice_model, move_batch
from rwkv_state_choice.losses import choice_cross_entropy
from rwkv_state_choice.model import RWKV7Backbone, StateChoiceMatcher


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def train(config: dict) -> None:
    training = config["training"]
    data_config = config["data"]
    model_config = config["model"]

    seed = int(training.get("seed", 42))
    _seed_everything(seed)

    device = torch.device(training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = getattr(torch, model_config.get("dtype", "bfloat16"))

    train_examples = load_mmlu_split(
        data_config["train_split"],
        dataset_id=data_config["dataset_id"],
        revision=data_config["revision"],
        max_examples=data_config.get("max_train_examples"),
    )
    validation_examples = load_mmlu_split(
        data_config["validation_split"],
        dataset_id=data_config["dataset_id"],
        revision=data_config["revision"],
    )
    if not train_examples:
        raise ValueError("training split is empty")

    backbone = RWKV7Backbone.from_pretrained(
        model_config["path"], dtype=dtype, device=device
    )
    collator = ChoiceCollator(
        backbone.tokenizer,
        max_problem_tokens=int(data_config["max_problem_tokens"]),
        max_candidate_tokens=int(data_config["max_candidate_tokens"]),
    )
    train_loader = DataLoader(
        train_examples,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_examples,
        batch_size=int(training.get("eval_batch_size", 1)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = StateChoiceMatcher(
        backbone,
        projection_dim=int(model_config.get("projection_dim", 512)),
        temperature=float(model_config.get("temperature", 0.07)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.initial_state.parameters(),
                "lr": float(training["state_learning_rate"]),
            },
            {
                "params": [
                    *model.query_projection.parameters(),
                    *model.candidate_projection.parameters(),
                ],
                "lr": float(training["projection_learning_rate"]),
            },
        ],
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config = output_dir / "config.yaml"
    shutil.copyfile(config["_config_path"], saved_config)

    epochs = int(training.get("epochs", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if epochs <= 0 or accumulation <= 0:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0
        total_batches = len(train_loader)
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")

        for step, cpu_batch in enumerate(progress):
            batch = move_batch(cpu_batch, device)
            scores = model(batch)["scores"]
            loss = choice_cross_entropy(scores, batch["targets"])

            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, total_batches - group_start)
            (loss / group_size).backward()

            batch_examples = int((batch["targets"] != -100).sum().item())
            total_loss += float(loss.detach()) * batch_examples
            total_examples += batch_examples

            end_of_group = (step + 1) % accumulation == 0 or step + 1 == total_batches
            if end_of_group:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(training.get("max_grad_norm", 1.0)),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        validation_metrics = evaluate_choice_model(model, validation_loader, device)
        state_rms = {
            "wkv": float(model.initial_state.wkv.detach().float().square().mean().sqrt()),
            "att_shift": float(model.initial_state.att_shift.detach().float().square().mean().sqrt()),
            "ffn_shift": float(model.initial_state.ffn_shift.detach().float().square().mean().sqrt()),
        }
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "train_examples": total_examples,
            "validation": validation_metrics,
            "initial_state_rms": state_rms,
        }
        print(json.dumps(epoch_metrics, sort_keys=True))

        save_adapter(
            output_dir / f"epoch-{epoch:03d}",
            model,
            {
                "base_model_repo": model_config["repo_id"],
                "base_model_revision": model_config["revision"],
                "dataset_id": data_config["dataset_id"],
                "dataset_revision": data_config["revision"],
                "train_split": data_config["train_split"],
                "epoch": epoch,
                "seed": seed,
                "metrics": epoch_metrics,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    config = _load_config(args.config)
    config["_config_path"] = str(Path(args.config).resolve())
    train(config)


if __name__ == "__main__":
    main()
