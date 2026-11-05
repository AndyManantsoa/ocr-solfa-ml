#!/usr/bin/env python3
"""MPS-only Phase 5B training. This command never reads the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sn_ml.train import TrainingConfig, run_training, validation_diagnostics, verify_real_training_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        verification = verify_real_training_step(root, "mps")
    except RuntimeError as error:
        raise SystemExit(f"MPS verification failed before full training: {error}") from error
    print("MPS training-step verification:", verification, flush=True)
    result = run_training(
        root,
        TrainingConfig(
            device="mps", batch_size=8, learning_rate=1e-3, epochs=50, patience=7,
            gradient_clip_norm=1.0, checkpoint_dir="checkpoints/phase5b_mps",
            metrics_path="exports/phase5b_mps_metrics.json",
        ),
        resume_path=args.resume,
        verbose=True,
    )
    best = min(result["history"], key=lambda row: (row["validation"]["cer"], -row["validation"]["exact_cell_accuracy"]))
    diagnostics = validation_diagnostics(root, root / result["best_checkpoint"], device_name="mps")
    diagnostics_path = root / "exports/phase5b_validation_diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"finished epochs={len(result['history'])} best_epoch={best['epoch']} best_cer={best['validation']['cer']:.4f}")
    print(f"best_checkpoint={result['best_checkpoint']}")
    print(f"validation_diagnostics={diagnostics_path.relative_to(root)}")


if __name__ == "__main__":
    main()
