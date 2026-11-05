"""Read-only dataset unification, validation, and leakage-safe split preparation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import random
import unicodedata
from typing import Any, Iterable, Mapping

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

from sn_ml.preprocessing import PreprocessConfig, pad_image_tensors, preprocess_image
from sn_ml.vocabulary import VOCAB
from sn_ml.vocabulary import encode_text

VISIBLE_CHARACTERS = frozenset(VOCAB)
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
# Lexical analysis only; this is deliberately not a musical grammar validator.
CHROMATIC_SYLLABLES = ("di", "de", "ri", "ra", "ma", "fi", "fe", "si", "se", "li", "le", "ta")
NATURAL_SYLLABLES = frozenset("drmfslt")


@dataclass(frozen=True)
class AdmissionPolicy:
    """Explicit, conservative rules for accepting labels into supervised splits."""

    name: str
    admitted_categories: frozenset[str]


REVIEWED_ONLY_POLICY = AdmissionPolicy(
    name="reviewed-only",
    admitted_categories=frozenset({"manually_corrected", "ocr_reviewed"}),
)
MANUAL_ONLY_POLICY = AdmissionPolicy(
    name="manual-corrections-only",
    admitted_categories=frozenset({"manually_corrected"}),
)
ADMISSION_POLICIES = {policy.name: policy for policy in (REVIEWED_ONLY_POLICY, MANUAL_ONLY_POLICY)}
QUALITY_CATEGORIES = (
    "manually_corrected", "ocr_reviewed", "visually_transcribed", "automatically_accepted",
    "unverified", "invalid_or_unresolved",
)


@dataclass(frozen=True)
class DatasetSource:
    name: str
    root: Path


@dataclass
class UnifiedRecord:
    """A normalized label plus provenance needed to resolve its original image."""

    id: str
    image: str
    label: str | None
    source_dataset: str
    source_root: str
    source_id: str
    source_group: str | None
    provenance: str | None
    normalized_label: str | None = None
    normalization_status: str = "not_applicable"
    annotation_quality: str = "unclassified"
    grouping_kind: str = "none"
    grouping_confidence: str = "none"
    image_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    issues: list[str] = field(default_factory=list)
    trusted: bool = False

    def image_path(self, project_root: Path) -> Path:
        return project_root / self.source_root / self.image

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetFormatError(ValueError):
    """The outer labels export schema is not supported."""


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def project_sources(project_root: Path) -> tuple[DatasetSource, ...]:
    return (
        DatasetSource("solfanote_dataset_labeled", project_root / "data/solfanote_dataset_labeled"),
        DatasetSource("solfanote-training", project_root / "data/solfanote-training"),
    )


def _read_entries(source: DatasetSource) -> list[Mapping[str, Any]]:
    labels_path = source.root / "labels.json"
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"Cannot read {labels_path}: {error}") from error
    entries = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise DatasetFormatError(f"{labels_path} must contain an object with a list-valued 'labels' key")
    return entries


def _provenance(source_name: str, row: Mapping[str, Any]) -> str | None:
    value = row.get("labelSource") if source_name == "solfanote-training" else row.get("transcriptionMethod")
    return value if isinstance(value, str) else None


def _annotation_quality(source_name: str, row: Mapping[str, Any], label: str | None) -> str:
    """Classify provenance without claiming independent verification where absent."""
    if label is None:
        return "invalid_or_unresolved"
    if source_name == "solfanote-training":
        if row.get("labelSource") == "manual-correction":
            return "manually_corrected"
        if row.get("labelSource") == "ocr-reviewed":
            return "ocr_reviewed"
        if row.get("originalReviewState") == "recognized":
            return "automatically_accepted"
        return "unverified"
    if row.get("transcriptionMethod") in {"direct_visual_transcription", "visual_group_transcription"}:
        return "visually_transcribed"
    return "unverified"


def normalize_label_spacing(label: str | None) -> tuple[str | None, str]:
    """Remove proven-formatting spaces only; preserve uncertain whitespace verbatim.

    Internal ASCII spaces between visible vocabulary characters are formatting in
    these exports. Leading/trailing, non-ASCII, or whitespace-only labels are
    deliberately returned unchanged and marked ambiguous.
    """
    if label is None:
        return None, "not_applicable"
    if not any(character.isspace() for character in label):
        return label, "not_needed"
    if (
        label != label.strip()
        or not label.strip()
        or any(character.isspace() and character != " " for character in label)
    ):
        return label, "ambiguous_whitespace"
    normalized = label.replace(" ", "")
    if not normalized or set(normalized) - VISIBLE_CHARACTERS:
        return label, "ambiguous_whitespace"
    return normalized, "spaces_removed"


def load_unified_records(project_root: Path) -> list[UnifiedRecord]:
    """Load both exports using ``finalLabel`` only; source data are never written."""
    records: list[UnifiedRecord] = []
    seen_ids: set[str] = set()
    for source in project_sources(project_root):
        source_root = source.root.relative_to(project_root).as_posix()
        for row_index, row in enumerate(_read_entries(source)):
            source_id = row.get("id")
            image = row.get("image")
            label = row.get("finalLabel")
            if not isinstance(source_id, str) or not source_id:
                source_id = f"invalid-{row_index:06d}"
            record_id = source_id if source_id not in seen_ids else f"{source.name}:{source_id}"
            seen_ids.add(record_id)
            source_group = row.get("lineId") if isinstance(row.get("lineId"), str) else None
            normalized_label, normalization_status = normalize_label_spacing(label if label is None or isinstance(label, str) else None)
            records.append(UnifiedRecord(
                id=record_id,
                image=image if isinstance(image, str) else "",
                label=label if label is None or isinstance(label, str) else None,
                source_dataset=source.name,
                source_root=source_root,
                source_id=source_id,
                source_group=source_group,
                provenance=_provenance(source.name, row),
                normalized_label=normalized_label,
                normalization_status=normalization_status,
                annotation_quality=_annotation_quality(source.name, row, label if label is None or isinstance(label, str) else None),
                grouping_kind="line_id" if source_group else "none",
                grouping_confidence="partial" if source_group else "none",
            ))
    return records


def _append_issue(record: UnifiedRecord, issue: str) -> None:
    if issue not in record.issues:
        record.issues.append(issue)


def _validate_image(record: UnifiedRecord, project_root: Path) -> None:
    if not record.image or Path(record.image).is_absolute() or ".." in Path(record.image).parts:
        _append_issue(record, "invalid_image_path")
        return
    path = record.image_path(project_root)
    if not path.is_file():
        _append_issue(record, "missing_image")
        return
    record.image_sha256 = sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record.width, record.height = image.size
    except (OSError, UnidentifiedImageError):
        _append_issue(record, "unreadable_image")
        return
    if not record.width or not record.height:
        _append_issue(record, "invalid_image_dimensions")
    elif min(record.width, record.height) < 8 or record.width * record.height < 64:
        _append_issue(record, "extremely_small_image")
    elif max(record.width, record.height) > 4096 or record.width * record.height > 16_000_000:
        _append_issue(record, "unusually_large_image")


def _validate_label(record: UnifiedRecord) -> None:
    if record.label is None:
        _append_issue(record, "missing_or_malformed_label")
        return
    if record.label == "":
        _append_issue(record, "blank_cell")
        return
    if unicodedata.normalize("NFC", record.label) != record.label:
        _append_issue(record, "non_nfc_label")
    if record.normalization_status == "spaces_removed":
        _append_issue(record, "formatting_spaces_normalized")
    elif record.normalization_status == "ambiguous_whitespace":
        _append_issue(record, "ambiguous_whitespace")
    if record.normalized_label is None or set(record.normalized_label) - VISIBLE_CHARACTERS:
        _append_issue(record, "unsupported_characters")


def _validate_provenance(record: UnifiedRecord, policy: AdmissionPolicy) -> None:
    if record.annotation_quality != "invalid_or_unresolved" and record.annotation_quality not in policy.admitted_categories:
        _append_issue(record, "not_admitted_by_policy")


def validate_records(
    records: list[UnifiedRecord],
    project_root: Path,
    policy: AdmissionPolicy = REVIEWED_ONLY_POLICY,
) -> None:
    """Populate validation metadata in place without modifying source files."""
    for record in records:
        record.issues = []
        record.trusted = False
        record.normalized_label, record.normalization_status = normalize_label_spacing(record.label)
        _validate_label(record)
        _validate_image(record, project_root)
        _validate_provenance(record, policy)
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.image_sha256 and record.label is not None:
            labels_by_hash[record.image_sha256].add(record.label)
    hash_counts = Counter(record.image_sha256 for record in records if record.image_sha256)
    for record in records:
        if record.image_sha256 and hash_counts[record.image_sha256] > 1:
            _append_issue(record, "exact_duplicate_image")
        if record.image_sha256 and len(labels_by_hash[record.image_sha256]) > 1:
            _append_issue(record, "conflicting_duplicate_label")
    blockers = {
        "invalid_image_path", "missing_image", "unreadable_image", "invalid_image_dimensions",
        "missing_or_malformed_label", "unsupported_characters", "ambiguous_whitespace", "conflicting_duplicate_label",
        "not_admitted_by_policy",
    }
    for record in records:
        record.trusted = record.label is not None and not any(issue in blockers for issue in record.issues)


def _syllable_counts(labels: Iterable[str]) -> tuple[Counter[str], Counter[str]]:
    natural: Counter[str] = Counter()
    chromatic: Counter[str] = Counter()
    chromatics = sorted(CHROMATIC_SYLLABLES, key=len, reverse=True)
    for label in labels:
        index = 0
        while index < len(label):
            token = next((item for item in chromatics if label.startswith(item, index)), None)
            if token:
                chromatic[token] += 1
                index += len(token)
            elif label[index] in NATURAL_SYLLABLES:
                natural[label[index]] += 1
                index += 1
            else:
                index += 1
    return natural, chromatic


def _dimension_statistics(records: Iterable[UnifiedRecord]) -> dict[str, Any]:
    dimensions = [(record.width, record.height) for record in records if record.width and record.height]
    if not dimensions:
        return {"count": 0}
    def summary(values: list[float]) -> dict[str, float]:
        values = sorted(values)
        return {"min": values[0], "median": values[len(values) // 2], "max": values[-1]}
    return {
        "count": len(dimensions),
        "width": summary([width for width, _ in dimensions]),
        "height": summary([height for _, height in dimensions]),
        "area": summary([width * height for width, height in dimensions]),
        "aspect_ratio": summary([width / height for width, height in dimensions]),
    }


def _exclusion_breakdown(records: Iterable[UnifiedRecord]) -> dict[str, Any]:
    """Separate overlapping issue mentions from one primary reason per exclusion."""
    primary = Counter()
    recoverable = 0
    priorities = (
        "missing_image", "unreadable_image", "invalid_image_path", "missing_or_malformed_label",
        "unsupported_characters", "ambiguous_whitespace", "conflicting_duplicate_label",
        "not_admitted_by_policy",
    )
    for record in records:
        if record.trusted:
            continue
        reason = next((issue for issue in priorities if issue in record.issues), "other")
        primary[reason] += 1
        if reason == "not_admitted_by_policy" and record.normalized_label is not None:
            recoverable += 1
    return {
        "unique_excluded_records": sum(primary.values()),
        "primary_reason": dict(sorted(primary.items())),
        "potentially_recoverable_pending_review": recoverable,
    }


def build_report(records: list[UnifiedRecord]) -> dict[str, Any]:
    by_source: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_dataset].append(record)
    original_labels = [record.label for record in records if isinstance(record.label, str)]
    normalized_labels = [record.normalized_label for record in records if isinstance(record.normalized_label, str)]
    valid_labels = [label for label in normalized_labels if not set(label) - VISIBLE_CHARACTERS]
    natural, chromatic = _syllable_counts(valid_labels)
    issue_counts = Counter(issue for record in records for issue in record.issues)
    original_chars = Counter(character for label in original_labels for character in label)
    chars = Counter(character for label in normalized_labels for character in label)
    lengths = Counter(len(label) for label in normalized_labels)
    hashes: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for record in records:
        if record.image_sha256:
            hashes[record.image_sha256].append(record)
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    conflicts = [group for group in duplicate_groups if len({record.label for record in group}) > 1]
    issue_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for issue in record.issues:
            if len(issue_examples[issue]) < 20:
                issue_examples[issue].append({"id": record.id, "image": record.image, "source_dataset": record.source_dataset})
    source_counts = {}
    for source, source_records in sorted(by_source.items()):
        source_counts[source] = {
            "total": len(source_records),
            "valid_label_and_image": sum(not any(issue in record.issues for issue in ("missing_or_malformed_label", "unsupported_characters", "ambiguous_whitespace", "missing_image", "unreadable_image", "invalid_image_path")) for record in source_records),
            "trusted": sum(record.trusted for record in source_records),
            "blank_cells": sum(record.label == "" for record in source_records),
            "provenance": dict(sorted(Counter(record.provenance or "missing" for record in source_records).items())),
            "annotation_quality": dict(sorted(Counter(record.annotation_quality for record in source_records).items())),
            "image_dimensions": _dimension_statistics(source_records),
        }
    return {
        "schema_version": 1,
        "totals": {
            "records": len(records),
            "valid_label_and_image": sum(not any(issue in record.issues for issue in ("missing_or_malformed_label", "unsupported_characters", "ambiguous_whitespace", "missing_image", "unreadable_image", "invalid_image_path")) for record in records),
            "invalid_label_or_image": sum(any(issue in record.issues for issue in ("missing_or_malformed_label", "unsupported_characters", "ambiguous_whitespace", "missing_image", "unreadable_image", "invalid_image_path")) for record in records),
            "clean_without_flags": sum(not record.issues for record in records),
            "flagged": sum(bool(record.issues) for record in records),
            "trusted": sum(record.trusted for record in records),
            "blank_cells": sum(record.label == "" for record in records),
        },
        "per_dataset": source_counts,
        "issue_counts": dict(sorted(issue_counts.items())),
        "exclusion_breakdown": _exclusion_breakdown(records),
        "annotation_quality": {
            category: sum(record.annotation_quality == category for record in records)
            for category in QUALITY_CATEGORIES
        },
        "normalization": {
            "labels_with_original_spaces": sum(any(character.isspace() for character in label) for label in original_labels),
            "normalized_labels": sum(record.normalization_status == "spaces_removed" for record in records),
            "ambiguous_labels": sum(record.normalization_status == "ambiguous_whitespace" for record in records),
            "original_character_frequency": dict(sorted(original_chars.items())),
        },
        "character_frequency": dict(sorted(chars.items())),
        "unsupported_characters": dict(sorted((character, count) for character, count in chars.items() if character not in VISIBLE_CHARACTERS)),
        "whitespace": {"records_with_whitespace": sum(any(character.isspace() for character in label) for label in original_labels), "requires_vocabulary_change": False},
        "syllable_frequency": {"natural": dict(sorted(natural.items())), "chromatic": dict(sorted(chromatic.items()))},
        "label_length_distribution": {str(length): count for length, count in sorted(lengths.items())},
        "image_dimensions": _dimension_statistics(records),
        "duplicate_images": {"groups": len(duplicate_groups), "extra_records": sum(len(group) - 1 for group in duplicate_groups)},
        "conflicting_annotations": {"groups": len(conflicts), "records": sum(len(group) for group in conflicts)},
        "potential_class_imbalance": {
            "zero_frequency_vocabulary": sorted(VISIBLE_CHARACTERS - set(character for record in records if record.trusted for character in record.normalized_label or "")),
            "least_frequent_observed": dict(sorted(Counter(character for record in records if record.trusted for character in record.normalized_label or "").items(), key=lambda item: (item[1], item[0]))[:10]),
        },
        "issue_examples": dict(sorted(issue_examples.items())),
        "problem_samples": [{"id": record.id, "image": record.image, "source_dataset": record.source_dataset, "issues": record.issues} for record in records if record.issues][:100],
    }


def _connected_groups(records: list[UnifiedRecord]) -> list[list[UnifiedRecord]]:
    union_find = _UnionFind(record.id for record in records)
    first_by_source_group: dict[tuple[str, str], str] = {}
    first_by_hash: dict[str, str] = {}
    for record in records:
        if record.source_group:
            key = (record.source_dataset, record.source_group)
            if key in first_by_source_group:
                union_find.union(record.id, first_by_source_group[key])
            else:
                first_by_source_group[key] = record.id
        if record.image_sha256:
            if record.image_sha256 in first_by_hash:
                union_find.union(record.id, first_by_hash[record.image_sha256])
            else:
                first_by_hash[record.image_sha256] = record.id
    grouped: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for record in records:
        grouped[union_find.find(record.id)].append(record)
    return list(grouped.values())


def make_splits(
    records: list[UnifiedRecord],
    seed: int = 1337,
    locked_test_ids: Iterable[str] = (),
) -> dict[str, list[UnifiedRecord]]:
    """Deterministic connected-component splits; hashes and source groups cannot leak."""
    candidates = [record for record in records if record.trusted]
    groups = _connected_groups(candidates)
    locked_ids = set(locked_test_ids)
    total = len(candidates)
    target_sizes = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    all_chars = Counter(character for record in candidates for character in record.normalized_label or "")
    target_chars = {name: {character: count * SPLIT_RATIOS[name] for character, count in all_chars.items()} for name in SPLIT_NAMES}
    assigned: dict[str, list[list[UnifiedRecord]]] = {name: [] for name in SPLIT_NAMES}
    assigned_sizes: Counter[str] = Counter()
    assigned_chars = {name: Counter() for name in SPLIT_NAMES}
    locked_groups = [group for group in groups if any(record.id in locked_ids for record in group)]
    groups = [group for group in groups if group not in locked_groups]
    for group in locked_groups:
        assigned["test"].append(group)
        assigned_sizes["test"] += len(group)
        assigned_chars["test"].update(character for record in group for character in record.normalized_label or "")
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    for group in groups:
        group_chars = Counter(character for record in group for character in record.normalized_label or "")
        def cost(name: str) -> tuple[float, float, str]:
            current_size_error = ((assigned_sizes[name] - target_sizes[name]) / max(target_sizes[name], 1)) ** 2
            next_size_error = ((assigned_sizes[name] + len(group) - target_sizes[name]) / max(target_sizes[name], 1)) ** 2
            size_cost = next_size_error - current_size_error
            char_cost = sum(
                ((assigned_chars[name][character] + count - target_chars[name][character]) / max(target_chars[name][character], 1)) ** 2
                - ((assigned_chars[name][character] - target_chars[name][character]) / max(target_chars[name][character], 1)) ** 2
                for character, count in group_chars.items()
            )
            # Whole source-line groups are large relative to validation/test.
            # Keep requested split proportions primary; character cost is only
            # a tiny deterministic tie-breaker under this group constraint.
            return (size_cost + 0.0000001 * char_cost / max(len(group_chars), 1), assigned_sizes[name], name)
        destination = min(SPLIT_NAMES, key=cost)
        assigned[destination].append(group)
        assigned_sizes[destination] += len(group)
        assigned_chars[destination].update(group_chars)
    return {name: [record for group in assigned[name] for record in group] for name in SPLIT_NAMES}


def _shared_values(value_lists: Iterable[Iterable[str]]) -> list[str]:
    appearances: Counter[str] = Counter()
    for values in value_lists:
        appearances.update(set(values))
    return sorted(value for value, count in appearances.items() if count > 1)


def build_source_grouping_report(records: list[UnifiedRecord]) -> dict[str, Any]:
    """Report only identifiers actually present in the exports, not inferred IDs."""
    per_source: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for record in records:
        per_source[record.source_dataset].append(record)
    report = {}
    for source, source_records in sorted(per_source.items()):
        grouped = [record for record in source_records if record.source_group]
        report[source] = {
            "records": len(source_records),
            "records_with_reliable_group_field": len(grouped),
            "unique_groups": len({record.source_group for record in grouped}),
            "grouping_kind": "line_id" if grouped else "none",
            "confidence": "partial" if grouped else "none",
            "limitation": (
                "lineId keeps cells from the same score line together, but no hymn, sheet, or document identifier is exported."
                if grouped else
                "No hymn, sheet, document, collection-session, or source-line identifier is exported; sequential sample filenames do not encode one."
            ),
        }
    return {"schema_version": 1, "datasets": report}


def _review_tags(record: UnifiedRecord) -> list[str]:
    label = record.normalized_label or ""
    tags = []
    if any(note in label for note in CHROMATIC_SYLLABLES):
        tags.append("chromatic")
    if any(note in label for note in NATURAL_SYLLABLES):
        tags.append("natural")
    if "," in label or "'" in label:
        tags.append("octave_marker")
    if any(symbol in label for symbol in ".;-"):
        tags.append("rhythmic_punctuation")
    syllables = sum(label.count(note) for note in CHROMATIC_SYLLABLES) + sum(label.count(note) for note in NATURAL_SYLLABLES)
    if syllables > 1:
        tags.append("multiple_note_cell")
    if len(label) <= 2:
        tags.append("short_label")
    if len(label) >= 5:
        tags.append("long_label")
    if record.width and record.height:
        area = record.width * record.height
        tags.append("small_image" if area < 7_000 else "large_image" if area > 12_000 else "medium_image")
    return tags


def build_review_manifest(records: list[UnifiedRecord], seed: int = 1337, per_stratum: int = 8) -> dict[str, Any]:
    """Select reproducible, unpromoted review examples from the new export."""
    candidates = [
        record for record in records
        if record.source_dataset == "solfanote_dataset_labeled"
        and record.normalized_label is not None
        and not any(issue in record.issues for issue in ("missing_image", "unreadable_image", "invalid_image_path", "unsupported_characters", "ambiguous_whitespace"))
    ]
    strata: dict[str, list[UnifiedRecord]] = defaultdict(list)
    for record in candidates:
        for tag in _review_tags(record):
            strata[tag].append(record)
    selected: dict[str, UnifiedRecord] = {}
    stratum_ids: dict[str, list[str]] = {}
    for tag, records_for_tag in sorted(strata.items()):
        ordered = sorted(records_for_tag, key=lambda record: sha256(f"{seed}:{tag}:{record.id}".encode()).hexdigest())
        chosen = ordered[:per_stratum]
        stratum_ids[tag] = [record.id for record in chosen]
        selected.update({record.id: record for record in chosen})
    return {
        "schema_version": 1,
        "seed": seed,
        "purpose": "manual review only; inclusion does not change annotation reliability or admission status",
        "strata": stratum_ids,
        "records": [
            {
                "id": record.id,
                "image": record.image,
                "source_root": record.source_root,
                "original_label": record.label,
                "normalized_label": record.normalized_label,
                "normalization_status": record.normalization_status,
                "annotation_quality": record.annotation_quality,
                "provenance": record.provenance,
                "source_group": record.source_group,
                "image_width": record.width,
                "image_height": record.height,
                "review_tags": _review_tags(record),
            }
            for record in sorted(selected.values(), key=lambda record: record.id)
        ],
    }


def _existing_test_ids(path: Path, seed: int) -> set[str]:
    """Keep previously generated reviewed test records independent after refinement."""
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict) or payload.get("seed") != seed or not isinstance(payload.get("records"), list):
        return set()
    return {record.get("id") for record in payload["records"] if isinstance(record, dict) and isinstance(record.get("id"), str)}


def write_processed_outputs(
    project_root: Path,
    seed: int = 1337,
    policy: AdmissionPolicy = REVIEWED_ONLY_POLICY,
) -> dict[str, Any]:
    """Write derived artifacts under ``data/processed``; original datasets stay read-only."""
    records = load_unified_records(project_root)
    validate_records(records, project_root, policy=policy)
    report = build_report(records)
    output_root = project_root / "data/processed"
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    prior_test_ids = _existing_test_ids(output_root / "test.json", seed)
    splits = make_splits(records, seed=seed, locked_test_ids=prior_test_ids)
    def write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(output_root / "unified.json", {"schema_version": 1, "records": [record.to_dict() for record in records]})
    for name, split_records in splits.items():
        write_json(output_root / f"{name}.json", {"schema_version": 1, "seed": seed, "records": [record.to_dict() for record in split_records]})
    report["splits"] = {
        "seed": seed,
        "admission_policy": policy.name,
        "counts": {name: len(split_records) for name, split_records in splits.items()},
        "source_groups": {name: len(_connected_groups(split_records)) for name, split_records in splits.items()},
        "leakage_checks": {
            "shared_image_hashes": _shared_values([[record.image_sha256 for record in split_records if record.image_sha256] for split_records in splits.values()]),
            "shared_source_groups": _shared_values([[f"{record.source_dataset}:{record.source_group}" for record in split_records if record.source_group] for split_records in splits.values()]),
        },
        "preserved_prior_test_ids": len(prior_test_ids),
        "preserved_prior_test_ids_present": len(prior_test_ids & {record.id for record in splits["test"]}),
    }
    write_json(reports_root / "dataset_report.json", report)
    write_json(reports_root / "issues.json", {"issue_counts": report["issue_counts"], "examples": report["issue_examples"]})
    write_json(reports_root / "source_grouping_report.json", build_source_grouping_report(records))
    write_json(reports_root / "review_manifest.json", build_review_manifest(records, seed=seed))
    return report


def load_manifest(path: Path) -> list[UnifiedRecord]:
    """Load a generated split manifest without reopening either source labels file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise DatasetFormatError(f"{path} does not contain a records list")
    return [UnifiedRecord(**record) for record in records]


class OcrManifestDataset(Dataset[dict[str, Any]]):
    """Manifest-backed PyTorch dataset using normalized labels only as targets."""

    def __init__(
        self,
        records: list[UnifiedRecord],
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        *,
        training: bool = False,
    ) -> None:
        self.records = [record for record in records if record.trusted]
        self.project_root = project_root
        self.preprocess_config = preprocess_config
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.normalized_label is None:
            raise ValueError(f"Trusted record {record.id} has no normalized label")
        image, width = preprocess_image(
            record.image_path(self.project_root), self.preprocess_config, training=self.training
        )
        target = torch.tensor(encode_text(record.normalized_label), dtype=torch.long)
        return {"id": record.id, "image": image, "width": width, "target": target, "target_length": len(target)}


def collate_ocr_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-width images and concatenate CTC targets for a DataLoader."""
    if not samples:
        raise ValueError("Cannot collate an empty OCR batch")
    return {
        "ids": [sample["id"] for sample in samples],
        "images": pad_image_tensors([sample["image"] for sample in samples]),
        "widths": torch.tensor([sample["width"] for sample in samples], dtype=torch.long),
        "targets": torch.cat([sample["target"] for sample in samples]),
        "target_lengths": torch.tensor([sample["target_length"] for sample in samples], dtype=torch.long),
    }
