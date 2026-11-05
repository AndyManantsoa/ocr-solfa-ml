"""Synthetic CRNN/CTC and preprocessing checks; no OCR training occurs here."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from sn_ml.dataset import OcrManifestDataset, collate_ocr_batch, load_manifest
from sn_ml.inference import greedy_decode
from sn_ml.model import CRNNConfig, TonicSolfaCRNN, ctc_loss, select_device
from sn_ml.preprocessing import PreprocessConfig, pad_image_tensors, preprocess_image
from sn_ml.vocabulary import CTC_BLANK_INDEX, ctc_greedy_decode, decode_indices, encode_text


def test_vocabulary_and_ctc_repeated_character_decoding() -> None:
    assert decode_indices(encode_text("di.r")) == "di.r"
    assert ctc_greedy_decode([1, 1, CTC_BLANK_INDEX, 1, 8, 8]) == "ddi"


def test_preprocessing_preserves_variable_width_and_batches() -> None:
    config = PreprocessConfig(image_height=32, min_width=8, max_width=128)
    narrow, narrow_width = preprocess_image(Image.new("L", (12, 24), 255), config)
    wide, wide_width = preprocess_image(Image.new("L", (48, 24), 255), config)
    assert narrow.shape == (1, 32, narrow_width)
    assert wide_width > narrow_width
    batch = pad_image_tensors([narrow, wide])
    assert batch.shape == (2, 1, 32, wide_width)
    assert torch.isfinite(batch).all()


def test_model_forward_output_lengths_and_ctc_loss_are_stable() -> None:
    model = TonicSolfaCRNN(CRNNConfig(image_height=64))
    images = torch.rand(2, 1, 64, 96)
    logits = model(images)
    lengths = model.output_lengths(torch.tensor([64, 96]))
    assert logits.shape == (2, 24, 18)
    assert lengths.tolist() == [16, 24]
    loss = ctc_loss(logits, torch.tensor([1, 1, 8, 2]), lengths, torch.tensor([3, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.parameter_count < 500_000


def test_greedy_batch_decode_obeys_lengths() -> None:
    logits = torch.full((2, 4, 18), -10.0)
    logits[0, :, 1] = 10.0
    logits[1, 0, 1] = 10.0
    logits[1, 1, 0] = 10.0
    logits[1, 2, 8] = 10.0
    logits[1, 3, 1] = 10.0
    assert greedy_decode(logits, torch.tensor([4, 3])) == ["d", "di"]


def test_real_manifest_images_batch_and_forward() -> None:
    root = Path(__file__).resolve().parents[1]
    records = load_manifest(root / "data/processed/train.json")
    dataset = OcrManifestDataset(records[:2], root, PreprocessConfig(image_height=64))
    batch = collate_ocr_batch([dataset[0], dataset[1]])
    model = TonicSolfaCRNN()
    logits = model(batch["images"])
    assert logits.shape[0] == 2
    assert torch.all(model.output_lengths(batch["widths"]) <= logits.shape[1])


def test_mps_forward_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable in this runtime")
    model = TonicSolfaCRNN().to("mps")
    assert model(torch.rand(1, 1, 64, 64, device="mps")).device.type == "mps"
    assert select_device().type == "mps"
