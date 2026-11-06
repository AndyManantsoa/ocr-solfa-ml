"""Small, resumable CRNN/CTC training infrastructure for reviewed manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from sn_ml.dataset import OcrManifestDataset, collate_ocr_batch, load_manifest
from sn_ml.evaluate import RecognitionMetrics, alignment_errors, character_error_summary
from sn_ml.grammar import CHROMATIC_NOTES, NATURAL_NOTES, validate_tonic_solfa
from sn_ml.inference import greedy_decode
from sn_ml.model import CRNNConfig, TonicSolfaCRNN, ctc_loss, select_device
from sn_ml.preprocessing import PreprocessConfig, pad_image_tensors, preprocess_image
from sn_ml.vocabulary import VOCAB


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1337
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    epochs: int = 2
    patience: int = 3
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    num_workers: int = 0
    device: str = "auto"
    checkpoint_dir: str = "checkpoints/phase5a_smoke"
    metrics_path: str = "exports/phase5a_smoke_metrics.json"


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return select_device()
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; refusing a silent CPU fallback")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing a silent CPU fallback")
    return device


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {**batch, "images": batch["images"].to(device), "widths": batch["widths"].to(device)}


def train_step(
    model: TonicSolfaCRNN,
    batch: dict[str, Any],
    optimizer: AdamW,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    moved = _move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(moved["images"])
    loss = ctc_loss(logits, batch["targets"], model.output_lengths(moved["widths"]), batch["target_lengths"])
    if not torch.isfinite(loss):
        raise RuntimeError("CTC loss is non-finite")
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("Model gradients are non-finite")
    clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def validate_epoch(model: TonicSolfaCRNN, loader: DataLoader[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    metrics = RecognitionMetrics()
    for batch in loader:
        moved = _move_batch(batch, device)
        logits = model(moved["images"])
        loss = ctc_loss(logits, batch["targets"], model.output_lengths(moved["widths"]), batch["target_lengths"])
        total_loss += float(loss.detach().cpu())
        batch_count += 1
        predictions = greedy_decode(logits, model.output_lengths(moved["widths"]))
        offset = 0
        for length, prediction in zip(batch["target_lengths"].tolist(), predictions):
            reference = "".join(VOCAB[index - 1] for index in batch["targets"][offset:offset + length].tolist())
            metrics.add(reference, prediction)
            offset += length
    return {"loss": total_loss / max(batch_count, 1), **metrics.to_dict()}


def _manifest_loader(
    manifest_path: Path,
    project_root: Path,
    preprocess: PreprocessConfig,
    config: TrainingConfig,
    *,
    training: bool,
    limit: int | None,
) -> DataLoader[dict[str, Any]]:
    dataset = OcrManifestDataset(load_manifest(manifest_path), project_root, preprocess, training=training)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training, num_workers=config.num_workers, collate_fn=collate_ocr_batch)


def save_checkpoint(
    path: Path,
    model: TonicSolfaCRNN,
    optimizer: AdamW,
    epoch: int,
    history: list[dict[str, Any]],
    config: TrainingConfig,
    scheduler: ReduceLROnPlateau | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch,
        "history": history, "model_config": asdict(model.config), "training_config": asdict(config),
        "vocabulary": list(VOCAB),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }, path)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_checkpoint(
    path: Path,
    model: TonicSolfaCRNN,
    optimizer: AdamW | None = None,
    device: torch.device | None = None,
    scheduler: ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    if payload["vocabulary"] != list(VOCAB):
        raise ValueError("Checkpoint vocabulary does not match this project")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def run_training(
    project_root: Path,
    config: TrainingConfig = TrainingConfig(),
    *,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    resume_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train only train/val manifests and write best/latest checkpoints plus metrics."""
    set_reproducible_seed(config.seed)
    device = training_device(config.device)
    preprocess = PreprocessConfig()
    train_loader = _manifest_loader(project_root / "data/processed/train.json", project_root, preprocess, config, training=True, limit=train_limit)
    validation_loader = _manifest_loader(project_root / "data/processed/val.json", project_root, preprocess, config, training=False, limit=validation_limit)
    model = TonicSolfaCRNN(CRNNConfig()).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = (float("inf"), float("inf"))
    if resume_path:
        loaded = load_checkpoint(resume_path, model, optimizer, device, scheduler)
        history = list(loaded.get("history", []))
        start_epoch = int(loaded["epoch"]) + 1
        best_score = min(
            ((row["validation"]["cer"], -row["validation"]["exact_cell_accuracy"]) for row in history),
            default=(float("inf"), float("inf")),
        )
    checkpoint_dir = project_root / config.checkpoint_dir
    best_path, latest_path = checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        losses = [train_step(model, batch, optimizer, device, config.gradient_clip_norm) for batch in train_loader]
        validation = validate_epoch(model, validation_loader, device)
        scheduler.step(validation["cer"])
        row = {
            "epoch": epoch, "training_loss": sum(losses) / max(len(losses), 1), "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"], "duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        if verbose:
            print(
                f"epoch {epoch}/{config.epochs} train_loss={row['training_loss']:.4f} "
                f"val_loss={validation['loss']:.4f} cer={validation['cer']:.4f} "
                f"exact={validation['exact_cell_accuracy']:.4f} lr={row['learning_rate']:.2e} "
                f"seconds={row['duration_seconds']:.1f}",
                flush=True,
            )
        save_checkpoint(latest_path, model, optimizer, epoch, history, config, scheduler)
        score = (validation["cer"], -validation["exact_cell_accuracy"])
        if score < best_score:
            best_score = score
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, history, config, scheduler)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    result = {
        "device": device.type, "duration_seconds": time.perf_counter() - started,
        "train_samples": len(train_loader.dataset), "validation_samples": len(validation_loader.dataset),
        "history": history, "best_checkpoint": _display_path(best_path, project_root),
        "latest_checkpoint": _display_path(latest_path, project_root),
        "grammar_review_targets": sum(not validate_tonic_solfa(record.normalized_label or "").valid for record in load_manifest(project_root / "data/processed/train.json")),
    }
    metrics_path = project_root / config.metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_real_training_step(project_root: Path, device_name: str = "mps") -> dict[str, Any]:
    """Exercise one real manifest batch on an explicitly requested accelerator."""
    device = training_device(device_name)
    config = TrainingConfig(batch_size=8, device=device_name)
    loader = _manifest_loader(
        project_root / "data/processed/train.json", project_root, PreprocessConfig(), config,
        training=False, limit=config.batch_size,
    )
    batch = next(iter(loader))
    model = TonicSolfaCRNN(CRNNConfig()).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_step(model, batch, optimizer, device, config.gradient_clip_norm)
    updated = any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters()))
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if not updated or not finite_gradients:
        raise RuntimeError("Accelerator training-step verification did not update finite parameters")
    return {"device": device.type, "samples": len(batch["ids"]), "loss": loss, "parameters_updated": updated, "finite_gradients": bool(finite_gradients)}


@torch.no_grad()
def validation_diagnostics(
    project_root: Path,
    checkpoint_path: Path,
    *,
    device_name: str = "mps",
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Inspect representative validation predictions without applying grammar correction."""
    device = training_device(device_name)
    config = TrainingConfig(batch_size=8, device=device_name)
    loader = _manifest_loader(
        project_root / "data/processed/val.json", project_root, PreprocessConfig(), config,
        training=False, limit=limit,
    )
    model = TonicSolfaCRNN(CRNNConfig()).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    diagnostics: list[dict[str, Any]] = []
    for batch in loader:
        moved = _move_batch(batch, device)
        logits = model(moved["images"])
        predictions = greedy_decode(logits, model.output_lengths(moved["widths"]))
        offset = 0
        for record_id, length, prediction in zip(batch["ids"], batch["target_lengths"].tolist(), predictions):
            reference = "".join(VOCAB[index - 1] for index in batch["targets"][offset:offset + length].tolist())
            diagnostics.append({"id": record_id, **character_error_summary(reference, prediction)})
            offset += length
    return diagnostics


def _error_categories(reference: str, prediction: str) -> list[str]:
    text = reference + prediction
    categories = []
    if "," in text or "'" in text:
        categories.append("octave_marker")
    if any(symbol in text for symbol in ".;-"):
        categories.append("rhythmic_symbol")
    if any(token in text for token in CHROMATIC_NOTES):
        categories.append("chromatic_note")
    if any(left == right for left, right in zip(text, text[1:])):
        categories.append("repeated_character")
    parsed = validate_tonic_solfa(reference)
    if len(parsed.note_tokens) > 1 or sum(character in NATURAL_NOTES for character in reference) > 1:
        categories.append("multiple_note_cell")
    if reference == "" or prediction == "":
        categories.append("blank_cell")
    if ":" in text or "|" in text:
        categories.append("visual_beat_separator")
    return categories


@torch.no_grad()
def full_split_diagnostics(
    project_root: Path,
    checkpoint_path: Path,
    *,
    split_name: str,
    device_name: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every record in one existing manifest without changing it."""
    if split_name not in {"val", "test"}:
        raise ValueError("Full diagnostics supports only the fixed val or test manifest")
    device = training_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TonicSolfaCRNN(CRNNConfig(**checkpoint["model_config"])).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    loader = _manifest_loader(
        project_root / f"data/processed/{split_name}.json", project_root, PreprocessConfig(),
        TrainingConfig(batch_size=8, device=device_name), training=False, limit=None,
    )
    model.eval()
    metrics = RecognitionMetrics()
    category_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for batch in loader:
        moved = _move_batch(batch, device)
        logits = model(moved["images"])
        predictions = greedy_decode(logits, model.output_lengths(moved["widths"]))
        offset = 0
        for record_id, length, prediction in zip(batch["ids"], batch["target_lengths"].tolist(), predictions):
            reference = "".join(VOCAB[index - 1] for index in batch["targets"][offset:offset + length].tolist())
            offset += length
            metrics.add(reference, prediction)
            alignment = alignment_errors(reference, prediction)
            categories = _error_categories(reference, prediction) if alignment["operations"] else []
            if alignment["operations"]:
                category_counts.update(categories)
                error_counts.update({kind: alignment[kind + "s"] for kind in ("insertion", "deletion", "substitution")})
            records.append({
                "id": record_id,
                "reference": reference,
                "prediction": prediction,
                "character_errors": alignment["insertions"] + alignment["deletions"] + alignment["substitutions"],
                "grammar_validation": validate_tonic_solfa(prediction).to_dict(),
                "error_operations": alignment["operations"],
                "error_categories": categories,
            })
    summary = {
        **metrics.to_dict(),
        "error_operations": dict(error_counts),
        "error_categories": {
            category: category_counts[category]
            for category in (
                "octave_marker", "rhythmic_symbol", "chromatic_note", "repeated_character",
                "multiple_note_cell", "blank_cell", "visual_beat_separator",
            )
        },
        "incorrect_samples": sum(record["comparison_available"] and record["character_errors"] > 0 for record in records),
        "correct_samples": sum(record["comparison_available"] and record["character_errors"] == 0 for record in records),
    }
    return records, summary


def full_validation_diagnostics(
    project_root: Path, checkpoint_path: Path, *, device_name: str = "auto"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return full_split_diagnostics(project_root, checkpoint_path, split_name="val", device_name=device_name)


def full_test_diagnostics(
    project_root: Path, checkpoint_path: Path, *, device_name: str = "auto"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return full_split_diagnostics(project_root, checkpoint_path, split_name="test", device_name=device_name)


def verify_test_split_integrity(project_root: Path, evaluated_ids: set[str]) -> dict[str, Any]:
    """Prove that the evaluated IDs are exactly test-only and duplicate-safe."""
    splits = {name: load_manifest(project_root / f"data/processed/{name}.json") for name in ("train", "val", "test")}
    ids = {name: {record.id for record in records} for name, records in splits.items()}
    test_hashes = {record.image_sha256 for record in splits["test"] if record.image_sha256}
    test_groups = {f"{record.source_dataset}:{record.source_group}" for record in splits["test"] if record.source_group}
    other_hashes = {
        name: {record.image_sha256 for record in records if record.image_sha256}
        for name, records in splits.items() if name != "test"
    }
    other_groups = {
        name: {f"{record.source_dataset}:{record.source_group}" for record in records if record.source_group}
        for name, records in splits.items() if name != "test"
    }
    report = {
        "expected_test_samples": len(splits["test"]),
        "distinct_test_ids": len(ids["test"]),
        "evaluated_samples": len(evaluated_ids),
        "evaluated_ids_match_test": evaluated_ids == ids["test"],
        "id_overlap": {name: sorted(ids["test"] & ids[name]) for name in ("train", "val")},
        "shared_exact_image_hashes": {name: sorted(test_hashes & other_hashes[name]) for name in ("train", "val")},
        "shared_source_groups": {name: sorted(test_groups & other_groups[name]) for name in ("train", "val")},
    }
    if not report["evaluated_ids_match_test"] or any(report["id_overlap"].values()) or any(report["shared_exact_image_hashes"].values()) or any(report["shared_source_groups"].values()):
        raise RuntimeError("Held-out test split integrity check failed")
    return report


@torch.no_grad()
def evaluate_external_labeled_dataset(
    dataset_root: Path,
    checkpoint_path: Path,
    *,
    device_name: str = "auto",
    batch_size: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read-only inference for a separately supplied labels.json dataset.

    This deliberately does not route through dataset validation, admission, or
    split logic. Labels are compared exactly as supplied in ``finalLabel``.
    """
    payload = json.loads((dataset_root / "labels.json").read_text(encoding="utf-8"))
    rows = payload["labels"]
    device = training_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TonicSolfaCRNN(CRNNConfig(**checkpoint["model_config"])).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    metrics = RecognitionMetrics()
    category_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    unscored_ids: list[str] = []
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        images_and_widths = [preprocess_image(dataset_root / row["image"], PreprocessConfig(), training=False) for row in batch_rows]
        images = pad_image_tensors([image for image, _ in images_and_widths]).to(device)
        widths = torch.tensor([width for _, width in images_and_widths], dtype=torch.long, device=device)
        logits = model(images)
        predictions = greedy_decode(logits, model.output_lengths(widths))
        for row, prediction in zip(batch_rows, predictions):
            reference = row["finalLabel"]
            if reference is None:
                unscored_ids.append(row.get("id", ""))
                records.append({
                    "id": row.get("id"), "image": row["image"], "reference": None, "prediction": prediction,
                    "comparison_available": False, "character_errors": None,
                    "grammar_validation": validate_tonic_solfa(prediction).to_dict(),
                    "error_operations": [], "error_categories": [],
                })
                continue
            metrics.add(reference, prediction)
            alignment = alignment_errors(reference, prediction)
            categories = _error_categories(reference, prediction) if alignment["operations"] else []
            if alignment["operations"]:
                category_counts.update(categories)
                error_counts.update({kind: alignment[kind + "s"] for kind in ("insertion", "deletion", "substitution")})
            records.append({
                "id": row.get("id"), "image": row["image"], "reference": reference, "prediction": prediction,
                "comparison_available": True,
                "character_errors": alignment["insertions"] + alignment["deletions"] + alignment["substitutions"],
                "grammar_validation": validate_tonic_solfa(prediction).to_dict(),
                "error_operations": alignment["operations"], "error_categories": categories,
            })
    summary = {
        **metrics.to_dict(),
        "error_operations": dict(error_counts),
        "error_categories": {
            category: category_counts[category]
            for category in ("octave_marker", "rhythmic_symbol", "chromatic_note", "repeated_character", "multiple_note_cell", "blank_cell", "visual_beat_separator")
        },
        "incorrect_samples": sum(record["comparison_available"] and record["character_errors"] > 0 for record in records),
        "correct_samples": sum(record["comparison_available"] and record["character_errors"] == 0 for record in records),
        "total_images": len(records),
        "scored_samples": metrics.samples,
        "unscored_null_label_samples": len(unscored_ids),
        "unscored_null_label_ids": unscored_ids,
    }
    return records, summary
