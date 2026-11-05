"""Focused Phase 2 tests using tiny disposable exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from sn_ml.dataset import (
    MANUAL_ONLY_POLICY,
    REVIEWED_ONLY_POLICY,
    UnifiedRecord,
    build_source_grouping_report,
    build_review_manifest,
    load_unified_records,
    make_splits,
    normalize_label_spacing,
    validate_records,
    write_processed_outputs,
)


def _write_png(path: Path, color: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (24, 12), color=color).save(path)


def _write_export(root: Path, name: str, rows: list[dict]) -> None:
    export = root / "data" / name
    (export / "images").mkdir(parents=True, exist_ok=True)
    (export / "labels.json").write_text(json.dumps({"version": 1, "labels": rows}), encoding="utf-8")


def _sources(tmp_path: Path) -> None:
    _write_export(tmp_path, "solfanote_dataset_labeled", [
        {"id": "sample_00001", "image": "images/a.png", "finalLabel": "d", "reviewStatus": "labeled", "transcriptionMethod": "direct_visual_transcription"},
    ])
    _write_export(tmp_path, "solfanote-training", [
        {"id": "train-1", "lineId": "line-a", "image": "images/b.png", "rawOcr": "x", "normalizedOcr": "x", "finalLabel": "d.r", "labelSource": "manual-correction"},
        {"id": "train-2", "lineId": "line-a", "image": "images/c.png", "finalLabel": "", "labelSource": "ocr-reviewed"},
    ])
    _write_png(tmp_path / "data/solfanote_dataset_labeled/images/a.png", color=128)
    _write_png(tmp_path / "data/solfanote-training/images/b.png")
    _write_png(tmp_path / "data/solfanote-training/images/c.png", color=0)


def test_loads_both_formats_and_uses_final_label(tmp_path: Path) -> None:
    _sources(tmp_path)
    records = load_unified_records(tmp_path)
    assert [record.label for record in records] == ["d", "d.r", ""]
    assert records[1].image_path(tmp_path).is_file()
    assert records[1].source_group == "line-a"
    assert records[1].id == "train-1"


def test_validation_normalizes_formatting_spaces_without_rewriting_original_label(tmp_path: Path) -> None:
    _sources(tmp_path)
    records = load_unified_records(tmp_path)
    records[1].label = "d r"
    original = records[1].label
    validate_records(records, tmp_path)
    assert original == records[1].label
    assert records[1].normalized_label == "dr"
    assert "formatting_spaces_normalized" in records[1].issues
    assert "blank_cell" in records[2].issues
    assert records[2].trusted is True


def test_space_normalization_handles_chromatics_octaves_and_punctuation() -> None:
    assert normalize_label_spacing("d r") == ("dr", "spaces_removed")
    assert normalize_label_spacing("di r") == ("dir", "spaces_removed")
    assert normalize_label_spacing("d, r'") == ("d,r'", "spaces_removed")
    assert normalize_label_spacing("d ., r") == ("d.,r", "spaces_removed")
    assert normalize_label_spacing(" d r") == (" d r", "ambiguous_whitespace")
    assert normalize_label_spacing("d\tr") == ("d\tr", "ambiguous_whitespace")


def test_annotation_quality_and_policy_are_separate_from_original_labels(tmp_path: Path) -> None:
    _sources(tmp_path)
    records = load_unified_records(tmp_path)
    validate_records(records, tmp_path, policy=REVIEWED_ONLY_POLICY)
    assert records[0].annotation_quality == "visually_transcribed"
    assert records[0].trusted is False
    assert records[1].annotation_quality == "manually_corrected"
    assert records[1].trusted is True
    validate_records(records, tmp_path, policy=MANUAL_ONLY_POLICY)
    assert records[2].annotation_quality == "ocr_reviewed"
    assert records[2].trusted is False


def test_duplicate_and_conflicting_labels_are_detected(tmp_path: Path) -> None:
    _sources(tmp_path)
    duplicate = tmp_path / "data/solfanote-training/images/c.png"
    duplicate.write_bytes((tmp_path / "data/solfanote-training/images/b.png").read_bytes())
    records = load_unified_records(tmp_path)
    validate_records(records, tmp_path)
    assert all("exact_duplicate_image" in record.issues for record in records[1:])
    assert all("conflicting_duplicate_label" in record.issues for record in records[1:])
    assert not records[1].trusted


def _trusted(record_id: str, group: str, image_hash: str) -> UnifiedRecord:
    return UnifiedRecord(
        id=record_id, image=f"images/{record_id}.png", label="d", source_dataset="solfanote-training",
        source_root="data/solfanote-training", source_id=record_id, source_group=group,
        provenance="manual-correction", image_sha256=image_hash, trusted=True,
    )


def test_splits_are_reproducible_and_prevent_group_or_hash_leakage() -> None:
    records = [_trusted(f"a-{index}", "line-a", "same-hash") for index in range(5)]
    records += [_trusted(f"b-{index}", "line-b", f"hash-b-{index}") for index in range(5)]
    records += [_trusted(f"c-{index}", "line-c", f"hash-c-{index}") for index in range(5)]
    first = make_splits(records, seed=7)
    second = make_splits(records, seed=7)
    assert {name: [record.id for record in split] for name, split in first.items()} == {name: [record.id for record in split] for name, split in second.items()}
    locations = {record.id: name for name, split in first.items() for record in split}
    assert len({locations[f"a-{index}"] for index in range(5)}) == 1
    assert len({locations[f"b-{index}"] for index in range(5)}) == 1


def test_preserved_test_ids_keep_their_connected_group_in_test() -> None:
    records = [_trusted(f"a-{index}", "line-a", f"hash-a-{index}") for index in range(4)]
    records += [_trusted(f"b-{index}", "line-b", f"hash-b-{index}") for index in range(4)]
    splits = make_splits(records, seed=7, locked_test_ids={"a-0"})
    assert {record.id for record in splits["test"]}.issuperset({f"a-{index}" for index in range(4)})


def test_review_manifest_is_reproducible_and_does_not_promote_samples(tmp_path: Path) -> None:
    _sources(tmp_path)
    records = load_unified_records(tmp_path)
    validate_records(records, tmp_path)
    first = build_review_manifest(records, seed=3)
    second = build_review_manifest(records, seed=3)
    assert first == second
    assert first["purpose"].startswith("manual review")
    assert records[0].trusted is False


def test_source_grouping_reports_only_exported_identifiers(tmp_path: Path) -> None:
    _sources(tmp_path)
    records = load_unified_records(tmp_path)
    report = build_source_grouping_report(records)
    assert report["datasets"]["solfanote-training"]["grouping_kind"] == "line_id"
    assert report["datasets"]["solfanote-training"]["confidence"] == "partial"
    assert report["datasets"]["solfanote_dataset_labeled"]["grouping_kind"] == "none"


def test_preparation_preserves_source_labels_file(tmp_path: Path) -> None:
    _sources(tmp_path)
    source = tmp_path / "data/solfanote-training/labels.json"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    write_processed_outputs(tmp_path, seed=9)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert (tmp_path / "data/processed/unified.json").is_file()
    assert (tmp_path / "data/processed/reports/dataset_report.json").is_file()
