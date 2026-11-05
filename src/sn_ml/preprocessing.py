"""Conservative Pillow preprocessing for variable-width tonic sol-fa cells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageEnhance
import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessConfig:
    image_height: int = 64
    min_width: int = 16
    max_width: int = 384
    normalize_ink: bool = True
    augment_rotation_degrees: float = 1.5
    augment_brightness: float = 0.10
    augment_noise_std: float = 0.01


def preprocess_image(
    source: str | Path | Image.Image,
    config: PreprocessConfig = PreprocessConfig(),
    *,
    training: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, int]:
    """Return a ``[1, H, W]`` ink tensor and its unpadded width.

    No thresholding or sharpening is used, so small dots, commas, apostrophes,
    and thin strokes remain represented by their original grayscale values.
    """
    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        with Image.open(source) as opened:
            image = opened.copy()
    image = image.convert("L")
    # Module-level RNG is seeded by the training infrastructure for reproducible
    # single-worker smoke/full runs; callers may inject a dedicated RNG in tests.
    random_source = rng or random
    if training:
        angle = random_source.uniform(-config.augment_rotation_degrees, config.augment_rotation_degrees)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
        brightness = 1 + random_source.uniform(-config.augment_brightness, config.augment_brightness)
        image = ImageEnhance.Brightness(image).enhance(brightness)
    width, height = image.size
    scaled_width = max(config.min_width, round(width * config.image_height / max(height, 1)))
    scaled_width = min(scaled_width, config.max_width)
    image = image.resize((scaled_width, config.image_height), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    if config.normalize_ink:
        tensor = 1.0 - tensor
    if training and config.augment_noise_std:
        tensor = (tensor + torch.randn_like(tensor) * config.augment_noise_std).clamp(0.0, 1.0)
    return tensor, scaled_width


def pad_image_tensors(images: list[torch.Tensor]) -> torch.Tensor:
    """Right-pad variable-width grayscale tensors with white-background zeros."""
    if not images:
        raise ValueError("Cannot pad an empty image batch")
    height = images[0].shape[1]
    if any(image.ndim != 3 or image.shape[0] != 1 or image.shape[1] != height for image in images):
        raise ValueError("Images must all have shape [1, common_height, width]")
    batch = torch.zeros((len(images), 1, height, max(image.shape[2] for image in images)), dtype=torch.float32)
    for index, image in enumerate(images):
        batch[index, :, :, :image.shape[2]] = image
    return batch
