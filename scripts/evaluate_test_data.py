#!/usr/bin/env python3
"""Final read-only evaluation on data/test-data; never trains or splits data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sn_ml.train import evaluate_external_labeled_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=Path("data/test-data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase3_mps/best.pt"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset = args.dataset if args.dataset.is_absolute() else root / args.dataset
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    records, summary = evaluate_external_labeled_dataset(dataset, checkpoint, device_name=args.device)
    full_path = root / "exports/phase3_test_full.json"
    report_path = root / "exports/phase3_test_summary.json"
    full_path.write_text(json.dumps({"dataset": str(dataset), "checkpoint": str(checkpoint), "records": records}, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps({"dataset": str(dataset), "checkpoint": str(checkpoint), "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(f"new images={summary['total_images']} scored={summary['scored_samples']} unscored_null_labels={summary['unscored_null_label_samples']} cer={summary['cer']:.4%} exact={summary['exact_cell_accuracy']:.2%}")
    for record in records:
        if record["character_errors"]:
            print(f"{record['image']}: expected={record['reference']!r} predicted={record['prediction']!r}")
    print(f"full_predictions={full_path.relative_to(root)}")
    print(f"report={report_path.relative_to(root)}")


if __name__ == "__main__":
    main()
