#!/usr/bin/env python3
"""Create derived, read-only-safe SN-ML dataset manifests and reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from sn_ml.dataset import ADMISSION_POLICIES, write_processed_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic split seed (default: 1337).")
    parser.add_argument("--admission-policy", choices=sorted(ADMISSION_POLICIES), default="reviewed-only")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = write_processed_outputs(
        args.project_root.resolve(), seed=args.seed, policy=ADMISSION_POLICIES[args.admission_policy]
    )
    print(f"Processed {report['totals']['records']} records; trusted={report['totals']['trusted']}")
    print("Splits:", report["splits"]["counts"])


if __name__ == "__main__":
    main()
