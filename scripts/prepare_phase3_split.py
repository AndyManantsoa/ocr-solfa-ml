#!/usr/bin/env python3
"""Validate data/training and create its deterministic 85/15 grouped split."""

from __future__ import annotations

import argparse
from pathlib import Path

from sn_ml.dataset import write_phase3_train_validation_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(write_phase3_train_validation_manifests(args.project_root.resolve(), seed=args.seed))


if __name__ == "__main__":
    main()
