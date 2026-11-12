#!/usr/bin/env python3
"""Build the one-time canonical training corpus without touching source/test data.

The generated ``data/training/labels.json`` deliberately contains only image
and label.  Audit, provenance, and grouping data are kept in the separate
``manifests/training_consolidation_v1.json`` sidecar.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image, UnidentifiedImageError

from sn_ml.dataset import VISIBLE_CHARACTERS, normalize_label_spacing


SOURCE_SPECS = (
    ("solfanote-training", "data/solfanote-training"),
    ("solfanote_dataset_labeled", "data/solfanote_dataset_labeled"),
    ("SolfaNote_4800_Labeled_Cells", "data/SolfaNote_4800_Labeled_Cells"),
)
NEW_SOURCE_PAGE_SIZE = 320
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def _safe_image_path(root: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return root / candidate


def _legacy_test_rows(project_root: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """Return legacy source identifiers and hashes reserved for the frozen test."""
    manifest_path = project_root / "data/processed/test.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["records"]
    identifiers: set[tuple[str, str]] = set()
    hashes: set[str] = set()
    for row in rows:
        identifiers.add((str(row["source_dataset"]), str(row["source_id"])))
        image_hash = row.get("image_sha256")
        if isinstance(image_hash, str):
            hashes.add(image_hash)
    return identifiers, hashes


def _test_hashes(test_root: Path) -> set[str]:
    return {_sha256(path) for path in test_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def _normalized_label(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    normalized, status = normalize_label_spacing(value)
    if status == "ambiguous_whitespace" or not normalized or set(normalized) - VISIBLE_CHARACTERS:
        return None
    return normalized


def _new_page_group(row_index: int, source_name: str, source_count: int) -> dict[str, str] | None:
    if source_name != "SolfaNote_4800_Labeled_Cells":
        return None
    if source_count != 4800:
        raise ValueError(f"Expected 4,800 new records, found {source_count}")
    return {"kind": "source_page", "id": f"page-{row_index // NEW_SOURCE_PAGE_SIZE + 1:02d}", "derivation": "README ordering: 320 cells per source page"}


def build(project_root: Path) -> dict[str, Any]:
    output_root = project_root / "data/training"
    manifest_path = project_root / "manifests/training_consolidation_v1.json"
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing consolidated dataset or manifest")

    legacy_ids, legacy_hashes = _legacy_test_rows(project_root)
    test_hashes = _test_hashes(project_root / "data/test-data")
    candidates: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    per_source: dict[str, dict[str, int]] = {}
    seen_hashes: dict[str, str] = {}

    for source_name, relative_root in SOURCE_SPECS:
        source_root = project_root / relative_root
        rows = json.loads((source_root / "labels.json").read_text(encoding="utf-8"))["labels"]
        source_counts = Counter()
        for row_index, row in enumerate(rows):
            source_counts["records"] += 1
            if not isinstance(row, dict):
                exclusions["malformed_record"] += 1
                source_counts["excluded"] += 1
                continue
            source_id = row.get("id")
            image_relative = row.get("image")
            normalized = _normalized_label(row.get("finalLabel"))
            if normalized is None:
                exclusions["invalid_or_empty_label"] += 1
                source_counts["excluded"] += 1
                continue
            image_path = _safe_image_path(source_root, image_relative)
            if image_path is None or not image_path.is_file():
                exclusions["missing_or_unsafe_image"] += 1
                source_counts["excluded"] += 1
                continue
            if not _image_is_readable(image_path):
                exclusions["unreadable_image"] += 1
                source_counts["excluded"] += 1
                continue
            image_hash = _sha256(image_path)
            if (source_name, str(source_id)) in legacy_ids or image_hash in legacy_hashes:
                exclusions["frozen_legacy_test"] += 1
                source_counts["excluded_legacy_test"] += 1
                continue
            if image_hash in test_hashes:
                exclusions["test_data_hash"] += 1
                source_counts["excluded_test_data"] += 1
                continue
            if image_hash in seen_hashes:
                exclusions["duplicate_image_hash"] += 1
                source_counts["deduplicated"] += 1
                continue
            group = _new_page_group(row_index, source_name, len(rows))
            if group is None and isinstance(row.get("lineId"), str):
                group = {"kind": "line_id", "id": row["lineId"], "derivation": "source metadata"}
            candidates.append({
                "source_dataset": source_name,
                "source_root": relative_root,
                "source_id": source_id if isinstance(source_id, str) else f"row-{row_index + 1:06d}",
                "source_image": image_relative,
                "source_label": row.get("finalLabel"),
                "normalized_label": normalized,
                "source_image_sha256": image_hash,
                "source_group": group,
                "image_path": image_path,
            })
            seen_hashes[image_hash] = source_name
            source_counts["included"] += 1
        per_source[source_name] = dict(source_counts)

    with TemporaryDirectory(prefix="sn-ml-consolidation-", dir=project_root / "data") as temp_dir:
        staging = Path(temp_dir) / "training"
        staging_images = staging / "images"
        staging_images.mkdir(parents=True)
        labels: list[dict[str, str]] = []
        records: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            filename = f"sample_{index:05d}.png"
            target = staging_images / filename
            shutil.copyfile(candidate.pop("image_path"), target)
            labels.append({"image": f"images/{filename}", "label": candidate["normalized_label"]})
            records.append({"canonical_id": f"training-{index:05d}", "image": f"images/{filename}", **candidate})
        (staging / "labels.json").write_text(json.dumps({"labels": labels}, indent=2) + "\n", encoding="utf-8")
        # Rename is atomic within data/ and never changes any source directory.
        staging.rename(output_root)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": "Provenance and grouping sidecar for data/training/labels.json",
        "canonical_labels": "data/training/labels.json",
        "legacy_test_manifest": "data/processed/test.json",
        "test_data_root": "data/test-data",
        "source_page_grouping": {"source": "SolfaNote_4800_Labeled_Cells", "pages": 15, "records_per_page": NEW_SOURCE_PAGE_SIZE},
        "counts": {"training_records": len(records), "source_records": per_source, "exclusions": dict(sorted(exclusions.items()))},
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = build(args.project_root.resolve())
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
