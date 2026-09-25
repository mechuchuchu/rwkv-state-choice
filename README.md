# RWKV-State-Choice

RWKV-State-Choice is an experimental framework for selecting an answer from a
variable number of candidates. It uses a frozen RWKV-7 language-model backbone
and learns a shared initial recurrent state plus two small projection heads.
The first experiment trains on MMLU's `auxiliary_train` configuration, selects
settings using MMLU `validation`, and keeps `test` for final evaluation.

This project scores complete candidate strings. It does not generate an answer,
fine-tune the RWKV-7 weights, or train a separate state for each question.

## How it works

For each example, the model encodes the problem from a learned initial RWKV
cache. It encodes each candidate independently from the backbone's default
zero state. The final non-padding hidden state from each sequence is projected
and normalized. A scaled dot product between the problem vector and each
candidate vector gives the candidate scores; cross entropy trains the correct
candidate score to be highest.

The RWKV-7 backbone weights stay frozen, but autograd runs through the problem
encoding path so the loss can update the learned initial cache. The candidate
encoding path is evaluated without storing a backbone gradient graph. Both
projection heads are trainable. The cache is shared across examples and reset
to the learned initial values for every problem.

The learned cache contains the RWKV-7 `WKV`, `ATT_SHIFT`, and `FFN_SHIFT` state
for every layer. For the configured 1.5B checkpoint, `WKV` is kept in FP32 and
the shift tensors are cast to the model activation dtype when a batch cache is
built. The training code intentionally uses ordinary padded batches; it does
not use packed sequence boundaries, gradient checkpointing, or per-example
state updates.

## Repository layout

```text
configs/mmlu.yaml                   Experiment and model settings
src/rwkv_state_choice/schema.py     Benchmark-independent ChoiceExample type
src/rwkv_state_choice/data/         MMLU loader and batch collator
src/rwkv_state_choice/model/        RWKV-7 backend, initial state, matcher
src/rwkv_state_choice/losses.py     Candidate cross-entropy
src/rwkv_state_choice/evaluation.py Accuracy metrics
src/rwkv_state_choice/checkpoint.py Adapter save/load
src/rwkv_state_choice/train.py     Training entry point
src/rwkv_state_choice/evaluate.py  Evaluation entry point
```

## Requirements

- Python 3.10 or newer.
- PyTorch 2.11 and a Transformers 5.x release compatible with the RWKV-7
  repository's remote model code.
- The RWKV-7 model snapshot downloaded locally. The default configuration
  expects it at `/workspace/RWKV7-G1j-1.5B-20260831`.
- Network access on the first data load, unless the pinned MMLU revision is
  already present in the Hugging Face datasets cache.
- A CUDA GPU is recommended. The starter configuration uses `bfloat16`, a
  micro-batch size of 1, and gradient accumulation. Adjust the settings for
  available memory.

The current Vast PyTorch image provides `/venv/main` and the project pins
dependencies compatible with that environment. From this repository:

```bash
source /venv/main/bin/activate
uv pip install -e .
```

Check the active environment and GPU before training:

```bash
python -c 'import torch, transformers, datasets; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("transformers", transformers.__version__, "datasets", datasets.__version__)'
nvidia-smi
```

The model loader reads only local files (`local_files_only=True`). If the model
directory is missing, download the model snapshot first and update
`model.path` in the YAML file. `model.repo_id` and `model.revision` record which
snapshot the experiment is intended to use; they do not cause the model loader
to download or verify the local files.

## Dataset and splits

The MMLU adapter pins the dataset to revision
[`c30699e8356da336a370243923dbaf21066bb9fe`](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe).
The dataset has an `all` configuration for the standard `dev`, `validation`,
and `test` splits, and a separate `auxiliary_train` configuration whose remote
split is named `train`. The code maps the local name `auxiliary_train` to that
Hub configuration and normalizes the examples to the project's common schema.
See the [MMLU dataset page](https://huggingface.co/datasets/cais/mmlu) for the
dataset card and source details.

The default experiment uses:

| Purpose | Local split name | Hugging Face configuration / split |
| --- | --- | --- |
| Training | `auxiliary_train` | `auxiliary_train` / `train` |
| Validation | `validation` | `all` / `validation` |
| Final evaluation | `test` | `all` / `test` |

`auxiliary_train` contains auxiliary multiple-choice training data; it is
distinct from MMLU's standard `dev` split. Validation is evaluated at the end
of every epoch. The training command does not load the test split. Run the
evaluation command on `test` after choosing a checkpoint.

The first training or evaluation run downloads the requested dataset split into
the Hugging Face cache. `load_mmlu_split()` materializes the selected split as
a Python list, so allow enough host RAM when loading the full auxiliary set.
For a short pipeline check, add `max_train_examples: 1000` under `data` in the
YAML file. Remove that option for the full training split. This limit selects
the first rows before the training DataLoader shuffles them.

## Train

From the repository root, run:

```bash
source /venv/main/bin/activate
python -m rwkv_state_choice.train --config configs/mmlu.yaml
```

The command loads the configured auxiliary training and validation splits,
loads the local RWKV-7 snapshot, trains for the configured number of epochs,
prints a JSON metrics record after each epoch, and saves one adapter directory
per epoch. The starter configuration writes to:

```text
runs/mmlu-global-state/
├── config.yaml
├── epoch-001/
│   ├── adapter.safetensors
│   └── metadata.json
└── ...
```

The run directory stores a copy of the YAML used for that run. Each epoch
record includes training loss, validation accuracy, accuracy by subject,
accuracy by candidate count, and RMS values for each learned state tensor.
The checkpoint contains the learned state and projection weights, not the
1.5B-parameter backbone or optimizer state. Training cannot be resumed from an
adapter checkpoint as a full optimizer-state resume.

The first run may take time to download and decode the dataset. The starter
configuration is an initial experiment, not a validated performance recipe.
No model or dataset download is triggered by installing the package.

## Evaluate

Evaluate an epoch checkpoint on the held-out test split:

```bash
python -m rwkv_state_choice.evaluate \
  --config configs/mmlu.yaml \
  --adapter runs/mmlu-global-state/epoch-001 \
  --split test
```

To inspect validation accuracy for a saved checkpoint, use `--split
validation`. Evaluation loads the same local backbone, restores the adapter,
checks that the adapter's recorded base-model revision matches the configured
revision, and prints JSON containing overall accuracy, example count,
per-subject accuracy, and accuracy by number of candidates. The evaluation
device is CUDA when available and CPU otherwise.

## Configuration

The starter file is [`configs/mmlu.yaml`](configs/mmlu.yaml). Important options:

| YAML key | Default | Meaning |
| --- | --- | --- |
| `model.path` | `/workspace/RWKV7-G1j-1.5B-20260831` | Local model snapshot directory used by Transformers. |
| `model.repo_id` | `RWKV/RWKV7-G1j-1.5B-20260831` | Model repository recorded in checkpoint metadata. |
| `model.revision` | `2c18b29ab7fbece25ff6112281eea0fa41fcb30f` | Intended model snapshot revision recorded in metadata and checked during evaluation. |
| `model.dtype` | `bfloat16` | Activation and backbone weight dtype requested when loading the model. WKV cache parameters remain FP32. |
| `model.projection_dim` | `512` | Output width of the problem and candidate projection heads. |
| `model.temperature` | `0.07` | Divisor applied to normalized dot-product scores before cross entropy. |
| `data.dataset_id` | `cais/mmlu` | Hugging Face dataset repository. |
| `data.revision` | `c30699e8356da336a370243923dbaf21066bb9fe` | Pinned dataset revision. |
| `data.train_split` | `auxiliary_train` | Training config/split loaded by the MMLU adapter. |
| `data.validation_split` | `validation` | Split evaluated after each training epoch. |
| `data.test_split` | `test` | Documents the intended final split; the evaluation CLI currently selects a split directly with `--split`. |
| `data.max_problem_tokens` | `256` | Maximum problem tokens; longer problems are truncated. |
| `data.max_candidate_tokens` | `128` | Maximum tokens per candidate; longer candidates are truncated. |
| `training.seed` | `42` | Python and PyTorch random seed. |
| `training.device` | `cuda` | Training device. Set to `cpu` to request CPU training. |
| `training.batch_size` | `1` | Number of examples in each forward pass. |
| `training.eval_batch_size` | `1` | Number of examples per validation/evaluation forward pass. |
| `training.gradient_accumulation_steps` | `16` | Micro-batches accumulated for one optimizer update. |
| `training.epochs` | `1` | Number of full passes over the loaded training examples. |
| `training.state_learning_rate` | `0.0001` | AdamW learning rate for the initial recurrent state. |
| `training.projection_learning_rate` | `0.0005` | AdamW learning rate for both projection heads. |
| `training.weight_decay` | `0.0` | AdamW weight decay for trainable parameters. |
| `training.max_grad_norm` | `1.0` | Gradient clipping threshold for trainable parameters. |
| `training.output_dir` | `runs/mmlu-global-state` | Directory for the config copy and epoch checkpoints. |

The loader also accepts optional `data.max_train_examples` for a deterministic
prefix subset of the training split. It is not set in the default YAML.

To use a smaller micro-batch, shorter sequences, or another output directory,
edit the YAML and then run the same command. Gradient accumulation changes the
effective update batch size without removing the activation memory required by
one micro-batch. If using CPU, set `training.device: cpu` and consider changing
`model.dtype` to `float32`; CPU speed and dtype support depend on the machine.

## Data format and extending the benchmark adapter

The model consumes benchmark-independent `ChoiceExample` objects:

```python
from rwkv_state_choice.schema import ChoiceExample

example = ChoiceExample(
    problem="Which number is prime?",
    candidates=["21", "23", "25", "27"],
    target=1,  # zero-based index into candidates; use None for inference-only data
    example_id="example-001",
    metadata={"subject": "arithmetic"},
)
```

Candidate order is significant: `target` is the zero-based index of the
correct candidate, and returned score position `i` corresponds to
`candidates[i]`. The MMLU adapter accepts integer answer indices or letter
labels such as `A`, `B`, `C`, and `D`, and converts them into candidate indices.
Benchmark details such as subject and source revision are retained in metadata
but are not model inputs.

To add another dataset, write an adapter that converts its rows into
`ChoiceExample` values. The collator tokenizes the problem and candidate text
separately, pads variable candidate counts within a batch, and returns tensors
with a candidate mask. The current command-line train/evaluate programs load
MMLU specifically; a new adapter must also be wired into those entry points (or
used from a custom training script) before it can be selected from YAML.

## Troubleshooting

- **Model directory not found:** download the expected RWKV-7 snapshot, or set
  `model.path` to the actual local directory. The current loader does not fetch
  model weights from the Hub.
- **CUDA was requested but is not available:** check `torch.cuda.is_available()`
  and set `training.device` to `cpu` only if CPU training is intended. For
  evaluation, the script selects CPU automatically when CUDA is unavailable.
- **Out of GPU memory:** reduce `training.batch_size`, `training.eval_batch_size`,
  or the two token limits. Gradient accumulation can preserve the number of
  examples per update after reducing the micro-batch size.
- **Dataset loading fails:** confirm that `datasets` is installed and that the
  instance can reach Hugging Face, or that the pinned revision is cached.
- **Adapter revision mismatch:** use the same configured base-model revision
  that was recorded when the adapter was trained. The model path must point to
  the corresponding local snapshot.
- **Candidate order mismatch:** keep candidates in a stable order and ensure
  each target is the zero-based index for that exact order.

## Current scope

The implemented experiment covers MMLU loading, the shared global initial
state, candidate scoring, epoch-level validation, and adapter checkpoints. It
does not yet include MMLU-Pro or MATH-500 adapters, prompt templates, language
generation, distributed training, optimizer-state resume, automatic Hub model
downloads, or a command-line inference interface for unlabeled custom data.
