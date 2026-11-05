"""Training infrastructure checks using synthetic CTC batches only."""

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from sn_ml.model import TonicSolfaCRNN, ctc_loss
from sn_ml.train import (
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_step,
    training_device,
    validate_epoch,
    run_training,
    verify_real_training_step,
)


def _batch() -> dict:
    return {
        "ids": ["one", "two"],
        "images": torch.rand(2, 1, 64, 64),
        "widths": torch.tensor([64, 64]),
        "targets": torch.tensor([1, 8, 2]),
        "target_lengths": torch.tensor([2, 1]),
    }


def test_ctc_loss_and_training_step_update_parameters() -> None:
    model = TonicSolfaCRNN()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    before = model.classifier[-1].weight.detach().clone()
    loss = train_step(model, _batch(), optimizer, torch.device("cpu"), 1.0)
    assert loss >= 0
    assert not torch.equal(before, model.classifier[-1].weight.detach())


def test_validation_metrics_calculate_without_training() -> None:
    metrics = validate_epoch(TonicSolfaCRNN(), [_batch()], torch.device("cpu"))
    assert metrics["loss"] >= 0
    assert 0 <= metrics["cer"]
    assert 0 <= metrics["exact_cell_accuracy"] <= 1
    assert 0 <= metrics["blank_prediction_frequency"] <= 1


def test_checkpoint_round_trip_and_resume_metadata(tmp_path: Path) -> None:
    model = TonicSolfaCRNN()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    history = [{"epoch": 1, "training_loss": 2.0, "validation": {"loss": 1.5}}]
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, 1, history, TrainingConfig())
    restored_model = TonicSolfaCRNN()
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    payload = load_checkpoint(path, restored_model, restored_optimizer, torch.device("cpu"))
    assert payload["epoch"] == 1
    assert payload["history"] == history
    assert torch.equal(model.classifier[-1].weight, restored_model.classifier[-1].weight)


def test_device_selection_refuses_silent_unavailable_accelerator_fallback() -> None:
    assert training_device("auto").type in {"cpu", "cuda", "mps"}
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS"):
            training_device("mps")


def test_run_training_resumes_from_latest_checkpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = TrainingConfig(
        batch_size=2,
        epochs=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    first = run_training(root, config, train_limit=2, validation_limit=2)
    resumed = run_training(root, TrainingConfig(
        batch_size=2,
        epochs=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        metrics_path=str(tmp_path / "metrics.json"),
    ), train_limit=2, validation_limit=2, resume_path=tmp_path / "checkpoints/latest.pt")
    assert first["history"][-1]["epoch"] == 1
    assert resumed["history"][-1]["epoch"] == 2
    assert len(resumed["history"]) == 2


def test_real_mps_training_step_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable in this runtime")
    result = verify_real_training_step(Path(__file__).resolve().parents[1], "mps")
    assert result["device"] == "mps"
    assert result["parameters_updated"] is True
    assert result["finite_gradients"] is True
