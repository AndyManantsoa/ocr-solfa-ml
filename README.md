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

`data/training/` is the canonical training corpus. `data/test-data/` is final
evaluation only and is never read by training or split preparation. The
consolidation provenance and source-page groups are retained in
`manifests/training_consolidation_v1.json`.

## Phase 3 workflow

```zsh
source .venv/bin/activate
python scripts/prepare_phase3_split.py --seed 1337
python scripts/run_smoke_training.py --train-limit 32 --validation-limit 16 --epochs 1
python scripts/run_full_training.py
```

The split script validates the canonical corpus and writes an 85/15 grouped
train/validation split under `manifests/`. Full training is MPS-only and never
loads `data/test-data/`.

## Grammar and model verification

`sn_ml.grammar` validates the closed tonic sol-fa syntax without changing a
prediction. `sn_ml.model` provides a compact CNN → bidirectional GRU → CTC
network, and `sn_ml.preprocessing` converts original cell images into
grayscale, aspect-ratio-preserving variable-width tensors. These components are
implemented and tested only; no model training is started in this project phase.

The smoke run writes only `checkpoints/phase3_smoke/` and
`exports/phase3_smoke_metrics.json`.
