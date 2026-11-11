"""Encode (B, 2, 256, 256) struct maps to (B, 2, 32, 32) on the protein AE latent grid.

Optionally loads pretrained weights (a state_dict or a checkpoint containing one).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class StructLatentEncoder(nn.Module):
    """Lightweight CNN: three stride-2 blocks downsample 256→32, 2→2 channels."""

    def __init__(self, in_ch: int = 2, out_ch: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_struct_encoder(
    ckpt_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    freeze: bool = False,
) -> StructLatentEncoder:
    """freeze=True: eval, no gradients. freeze=False: trainable.

    Random init requires freeze=False, otherwise the struct branch never updates.
    """
    enc = StructLatentEncoder()
    if ckpt_path and os.path.isfile(ckpt_path):
        blob: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        if not isinstance(sd, dict):
            raise ValueError(f"Could not parse a state_dict from {ckpt_path}")
        enc.load_state_dict(sd, strict=False)
        print(f"[StructLatentEncoder] loaded {ckpt_path} (strict=False)")
    else:
        if ckpt_path:
            print(f"[StructLatentEncoder] warning: file not found {ckpt_path}, using random init")
    enc.to(device=device, dtype=dtype)
    if freeze:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
    else:
        enc.train()
        for p in enc.parameters():
            p.requires_grad_(True)
    return enc
