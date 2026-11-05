"""CTC decoding helpers; grammar ranking remains a later integration step."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sn_ml.vocabulary import ctc_greedy_decode


@dataclass(frozen=True)
class DecodeCandidate:
    text: str
    source: str = "greedy_ctc"


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """Greedily decode ``[batch, time, classes]`` logits using valid time lengths."""
    classes = logits.argmax(dim=-1).detach().cpu()
    return [ctc_greedy_decode(row[: int(length)].tolist()) for row, length in zip(classes, lengths.cpu())]


def generate_candidates(logits: torch.Tensor, lengths: torch.Tensor) -> list[list[DecodeCandidate]]:
    """Stable candidate-generation interface for future beam/grammar ranking."""
    return [[DecodeCandidate(text)] for text in greedy_decode(logits, lengths)]
