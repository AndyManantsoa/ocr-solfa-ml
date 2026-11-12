#!/usr/bin/env python3
"""Predict one tonic sol-fa beat-cell image from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sn_ml.inference import greedy_decode
from sn_ml.model import CRNNConfig, TonicSolfaCRNN
from sn_ml.preprocessing import PreprocessConfig, preprocess_image
from sn_ml.train import load_checkpoint, training_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase3_mps/best.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = training_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TonicSolfaCRNN(CRNNConfig(**checkpoint["model_config"])).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    image, width = preprocess_image(args.image, PreprocessConfig(), training=False)
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    print(greedy_decode(logits, model.output_lengths(torch.tensor([width], device=device)))[0])


if __name__ == "__main__":
    main()
