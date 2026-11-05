# SN-ML

SN-ML is a standalone, lightweight research project for personalized OCR of
cropped tonic sol-fa beat cells. It will use a closed character vocabulary and
a CNN/CTC recognizer in later phases; this initialization contains no dataset
processing, model, training, inference, or musical-grammar implementation.

## Setup

This project requires Python 3.12. Create the environment and install the
package with its development dependencies:

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Apple Silicon, the PyTorch wheel can use MPS when it is available. CUDA and
CPU remain supported by PyTorch on their respective platforms.

## Verification

```zsh
source .venv/bin/activate
pytest
```

The two source datasets under `data/` are deliberately ignored by Git and are
not read or changed during this phase.

## Dataset preparation

Phase 2 reads both immutable dataset exports using `finalLabel` only and writes
derived manifests to `data/processed/`:

```zsh
source .venv/bin/activate
python scripts/prepare_dataset.py --seed 1337
```

Formatting-only internal spaces are retained in `label` and emitted separately
as `normalized_label`. The default `reviewed-only` policy admits only
`manual-correction` and `ocr-reviewed` labels; use `--admission-policy
manual-corrections-only` for a stricter derived split. The generated reports
record validation issues, duplicate hashes, provenance, source-grouping limits,
review examples, and split-leakage checks. The visually transcribed export
remains audit/review-only until independently verified.

## Grammar and model verification

`sn_ml.grammar` validates the closed tonic sol-fa syntax without changing a
prediction. `sn_ml.model` provides a compact CNN → bidirectional GRU → CTC
network, and `sn_ml.preprocessing` converts original cell images into
grayscale, aspect-ratio-preserving variable-width tensors. These components are
implemented and tested only; no model training is started in this project phase.

## Bounded smoke experiment

Phase 5A adds resumable AdamW/CTC training and validation infrastructure. The
following command runs only a small, two-epoch smoke experiment on train and
validation manifests; it never reads the held-out test manifest:

```zsh
python scripts/run_smoke_training.py --train-limit 32 --validation-limit 16
```

It writes best/latest checkpoints under `checkpoints/phase5a_smoke/` and JSON
metrics under `exports/phase5a_smoke_metrics.json`. A full training run still
requires an explicit later-phase decision.
