#!/usr/bin/env python3
"""Final read-only held-out test evaluation; never trains or selects a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sn_ml.train import full_test_diagnostics, verify_test_split_integrity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase5b_mps/best.pt"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    records, summary = full_test_diagnostics(root, checkpoint, device_name=args.device)
    integrity = verify_test_split_integrity(root, {record["id"] for record in records})
    full_path = root / "exports/phase5b_test_full.json"
    summary_path = root / "exports/phase5b_test_summary.json"
    full_path.write_text(json.dumps({"checkpoint": str(checkpoint.relative_to(root)), "records": records}, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps({"checkpoint": str(checkpoint.relative_to(root)), "split_integrity": integrity, **summary}, indent=2) + "\n", encoding="utf-8")
    print(f"test samples={summary['samples']} cer={summary['cer']:.4%} exact={summary['exact_cell_accuracy']:.2%}")
    print(f"errors: insertions={summary['error_operations'].get('insertion', 0)} deletions={summary['error_operations'].get('deletion', 0)} substitutions={summary['error_operations'].get('substitution', 0)}")
    for record in records:
        if record["character_errors"]:
            print(f"{record['id']}: ref={record['reference']!r} pred={record['prediction']!r} errors={record['character_errors']} categories={','.join(record['error_categories']) or 'none'}")
    print(f"split_integrity=passed distinct_test_ids={integrity['distinct_test_ids']}")
    print(f"full_predictions={full_path.relative_to(root)}")
    print(f"summary={summary_path.relative_to(root)}")


if __name__ == "__main__":
    main()
