#!/usr/bin/env python3
"""Evaluate a separate labeled dataset without validation, training, or split changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sn_ml.train import evaluate_external_labeled_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=Path("data/new labeled data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase5b_mps/best.pt"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset = args.dataset if args.dataset.is_absolute() else root / args.dataset
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    records, summary = evaluate_external_labeled_dataset(dataset, checkpoint, device_name=args.device)
    test_summary_path = root / "exports/phase5b_test_summary.json"
    previous_test = json.loads(test_summary_path.read_text(encoding="utf-8")) if test_summary_path.is_file() else None
    comparison = {
        "previous_held_out_test": {
            "samples": previous_test["samples"], "cer": previous_test["cer"], "exact_cell_accuracy": previous_test["exact_cell_accuracy"],
        } if previous_test else None,
        "new_real_world_dataset": {
            "samples": summary["samples"], "cer": summary["cer"], "exact_cell_accuracy": summary["exact_cell_accuracy"],
        },
    }
    full_path = root / "exports/new_labeled_data_evaluation_full.json"
    report_path = root / "exports/new_labeled_data_evaluation_report.json"
    full_path.write_text(json.dumps({"dataset": str(dataset), "checkpoint": str(checkpoint), "records": records}, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps({"dataset": str(dataset), "checkpoint": str(checkpoint), "summary": summary, "comparison": comparison}, indent=2) + "\n", encoding="utf-8")
    print(f"new images={summary['total_images']} scored={summary['scored_samples']} unscored_null_labels={summary['unscored_null_label_samples']} cer={summary['cer']:.4%} exact={summary['exact_cell_accuracy']:.2%}")
    if previous_test:
        print(f"previous test: samples={previous_test['samples']} cer={previous_test['cer']:.4%} exact={previous_test['exact_cell_accuracy']:.2%}")
    for record in records:
        if record["character_errors"]:
            print(f"{record['image']}: expected={record['reference']!r} predicted={record['prediction']!r}")
    print(f"full_predictions={full_path.relative_to(root)}")
    print(f"report={report_path.relative_to(root)}")


if __name__ == "__main__":
    main()
