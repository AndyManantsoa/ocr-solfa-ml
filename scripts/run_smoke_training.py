#!/usr/bin/env python3
"""Run the bounded Phase 5A CRNN/CTC smoke experiment; never evaluates test.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from sn_ml.train import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--validation-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--resume", type=Path, help="Resume from a latest checkpoint within the same experiment.")
    args = parser.parse_args()
    result = run_training(
        args.project_root.resolve(),
        TrainingConfig(batch_size=args.batch_size, epochs=args.epochs),
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        resume_path=args.resume,
    )
    print(f"device={result['device']} duration_seconds={result['duration_seconds']:.2f}")
    print(f"samples=train:{result['train_samples']} validation:{result['validation_samples']}")
    print(f"best_checkpoint={result['best_checkpoint']}")
    for epoch in result["history"]:
        print(f"epoch={epoch['epoch']} train_loss={epoch['training_loss']:.4f} val_loss={epoch['validation']['loss']:.4f}")


if __name__ == "__main__":
    main()
