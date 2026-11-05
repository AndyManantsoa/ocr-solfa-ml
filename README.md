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
