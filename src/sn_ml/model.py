"""A compact variable-width CRNN with CTC-compatible outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sn_ml.vocabulary import CTC_BLANK_INDEX, CTC_CLASS_COUNT


@dataclass(frozen=True)
class CRNNConfig:
    image_height: int = 64
    cnn_channels: tuple[int, int, int] = (16, 32, 64)
    recurrent_hidden_size: int = 64
    recurrent_layers: int = 1
    dropout: float = 0.10
    class_count: int = CTC_CLASS_COUNT


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TonicSolfaCRNN(nn.Module):
    """Small CNN -> horizontal sequence -> bidirectional GRU -> CTC logits."""

    horizontal_stride = 4

    def __init__(self, config: CRNNConfig = CRNNConfig()) -> None:
        super().__init__()
        self.config = config
        first, second, third = config.cnn_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, first, kernel_size=3, padding=1), nn.BatchNorm2d(first), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(first, second, kernel_size=3, padding=1), nn.BatchNorm2d(second), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(second, third, kernel_size=3, padding=1), nn.BatchNorm2d(third), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.encoder = nn.GRU(
            input_size=third,
            hidden_size=config.recurrent_hidden_size,
            num_layers=config.recurrent_layers,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.recurrent_hidden_size * 2, config.class_count),
        )

    def output_lengths(self, input_widths: torch.Tensor) -> torch.Tensor:
        """CNN time lengths after its two horizontal stride-2 max pools."""
        return torch.clamp(input_widths.to(dtype=torch.long) // self.horizontal_stride, min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return CTC logits with shape ``[batch, time, classes]``."""
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected image batch with shape [batch, 1, height, width]")
        features = self.height_pool(self.features(images)).squeeze(2).transpose(1, 2)
        encoded, _ = self.encoder(features)
        return self.classifier(encoded)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable CTC loss; MPS uses CPU CTCLoss while retaining autograd."""
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    if log_probs.device.type == "mps":
        log_probs = log_probs.cpu()
    return nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)(
        log_probs,
        targets.to(dtype=torch.long, device="cpu"),
        input_lengths.to(dtype=torch.long, device="cpu"),
        target_lengths.to(dtype=torch.long, device="cpu"),
    )
