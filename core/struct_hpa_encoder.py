"""HPA VAE utilities.

1. HPAStructEncoder — struct encoder for the nucleus + tubulin maps
   (B, 2, 256, 256) -> (B, 8, L, L), with L set by latent_size
   - latent_size=32: avg_pool(k=2) over the native 64×64 HPA latent, so the
     struct grid matches the main branch
   - latent_size=64: native 64×64 output

2. HPAMainAE — main-branch AE adapter
   Adapts diffusers.AutoencoderKL to ``.encode(x).sample()`` / ``.decode(z)``,
   matching encode_protein_latent / decode_protein_latent in train.py.

Both classes load the same CELL-Diff pretrained `hpa_pretrained.bin`.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_FALLBACK_HPA_CKPT = os.path.join(
    _REPO_ROOT,
    "pretrained",
    "hpa_pretrained.bin",
)
DEFAULT_HPA_CKPT = os.environ.get("HPA_VAE_PRETRAINED_CKPT", _DEFAULT_FALLBACK_HPA_CKPT)


def _build_hpa_vae() -> nn.Module:
    """Same architecture as HPAVaeBackend in scripts/eval_vae_cypm_recon.py."""
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(128, 256, 512),
        latent_channels=4,
        layers_per_block=1,
    )


def _load_hpa_weights(vae: nn.Module, ckpt_path: str, tag: str, verbose: bool):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"HPA VAE ckpt not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        blob = blob["model"]
    elif isinstance(blob, dict) and "module" in blob:
        blob = blob["module"]
    sd = {k[len("vae."):] if k.startswith("vae.") else k: v for k, v in blob.items()}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "dummy" not in k]
    unexpected = [k for k in unexpected if "dummy" not in k]
    if verbose:
        print(f"[{tag}] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")


class HPAStructEncoder(nn.Module):
    """(B, 2, 256, 256) in [-1, 1] -> (B, 8, 32, 32) latent on the DiT 32×32 grid.

    - Both input channels share one HPA VAE (weights shared, as in CELL-Diff)
    - posterior.mode() gives a deterministic latent (no noise sampling)
    - Channel concat: first 4 ch = nucleus latent, last 4 ch = tubulin latent
    - Spatial 64->32 via avg_pool2d(kernel=2, stride=2)

    The DiT side patchifies (B,8,L,L) into 256 struct tokens, concatenated with
    the 256 noise tokens of the main branch (seq ~= 512).
    """

    def __init__(
        self,
        ckpt_path: str = DEFAULT_HPA_CKPT,
        latent_size: int = 32,
        verbose: bool = True,
    ):
        super().__init__()
        self.latent_size = int(latent_size)
        assert self.latent_size in (32, 64), \
            f"HPAStructEncoder.latent_size must be 32 or 64, got {latent_size}"
        self.vae = _build_hpa_vae()
        _load_hpa_weights(self.vae, ckpt_path, tag="HPAStructEncoder", verbose=verbose)
        # always frozen
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_single(self, x_1ch: torch.Tensor) -> torch.Tensor:
        """(B, 1, 256, 256) -> (B, 4, 64, 64) via posterior.mode()."""
        return self.vae.encode(x_1ch).latent_dist.mode()

    def forward(self, x_2ch: torch.Tensor) -> torch.Tensor:
        """x_2ch: (B, 2, 256, 256) in [-1, 1]. Returns (B, 8, latent_size, latent_size)."""
        assert x_2ch.dim() == 4 and x_2ch.shape[1] == 2, \
            f"HPAStructEncoder expects (B, 2, H, W), got {tuple(x_2ch.shape)}"
        # eval() keeps the internal BN/GroupNorm numerics stable
        with torch.no_grad():
            z0 = self._encode_single(x_2ch[:, 0:1])  # nucleus
            z1 = self._encode_single(x_2ch[:, 1:2])  # tubulin
        z = torch.cat([z0, z1], dim=1)  # (B, 8, 64, 64)
        if z.shape[-1] != self.latent_size:
            factor = z.shape[-1] // self.latent_size
            if factor <= 0:
                raise ValueError(
                    f"HPA latent spatial size {z.shape[-1]} is smaller than the requested {self.latent_size};"
                    "up-sampling is not implemented"
                )
            z = F.avg_pool2d(z, kernel_size=factor, stride=factor)
        return z  # (B, 8, latent_size, latent_size)

    def train(self, mode: bool = True):
        # always stay in eval; train(True) is ignored
        return super().train(False)


def load_hpa_struct_encoder(
    ckpt_path: str,
    device: torch.device,
    latent_size: int = 32,
    verbose: bool = True,
) -> HPAStructEncoder:
    enc = HPAStructEncoder(
        ckpt_path=ckpt_path or DEFAULT_HPA_CKPT,
        latent_size=latent_size,
        verbose=verbose,
    )
    enc.to(device)
    enc.eval()
    return enc


def hpa_struct_out_channels() -> int:
    """HPAStructEncoder output channels (4 nucleus + 4 tubulin)."""
    return 8


class _DiagonalGaussianAdapter:
    """Thin wrapper over diffusers DiagonalGaussianDistribution for ``.encode(...).sample()``."""

    def __init__(self, dist):
        self._dist = dist

    def sample(self):
        return self._dist.sample()

    def mode(self):
        return self._dist.mode()


class HPAMainAE(nn.Module):
    """HPA VAE (CELL-Diff ``hpa_pretrained.bin``) for the train.py main branch.

      - encode(x): returns a wrapper exposing ``.sample()`` / ``.mode()``
      - decode(z): returns a ``(B,1,H,W)`` tensor
      - setgpu(device): move to device and run in eval mode
    Latent: (B, 1, 256, 256) -> (B, 4, 64, 64).
    """

    def __init__(self, ckpt_path: str = DEFAULT_HPA_CKPT, verbose: bool = True):
        super().__init__()
        self.vae = _build_hpa_vae()
        _load_hpa_weights(self.vae, ckpt_path, tag="HPAMainAE", verbose=verbose)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self._device = torch.device("cpu")

    def encode(self, x: torch.Tensor) -> _DiagonalGaussianAdapter:
        out = self.vae.encode(x)
        return _DiagonalGaussianAdapter(out.latent_dist)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).sample

    def setgpu(self, device: torch.device = None):
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self._device)
        return self

    def train(self, mode: bool = True):
        # always stay in eval
        return super().train(False)


def load_hpa_main_ae(ckpt_path: str, device: torch.device, verbose: bool = True) -> HPAMainAE:
    ae = HPAMainAE(ckpt_path=ckpt_path or DEFAULT_HPA_CKPT, verbose=verbose)
    ae.setgpu(device)
    ae.eval()
    return ae


def hpa_main_latent_channels() -> int:
    """Latent channels of the HPA VAE main branch."""
    return 4


def hpa_main_latent_size(img_size: int = 256) -> int:
    """Latent spatial size of the HPA VAE for img_size (4x downsampling)."""
    return img_size // 4
