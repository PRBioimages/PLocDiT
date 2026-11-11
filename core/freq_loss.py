"""Frequency-domain reconstruction losses.

- ``mean_flat_freq_loss`` / ``freq_loss_magnitude``: circular low/high-band weighted
  MSE on the FFT magnitude spectrum.
- ``mean_flat_laplacian_loss``: 3x3 Laplacian edge loss (L1/L2), more local and
  sensitive to punctate / membrane structures.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Cache circular low-freq masks by (H, W, ratio) instead of rebuilding each step
_MASK_CACHE: dict[tuple[int, int, float], torch.Tensor] = {}


def _radial_distance_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Euclidean distance from the fftshifted spectrum centre to each bin, (H, W)."""
    # fftshift places DC at h//2; using (h-1)/2 shifts the mask by half a pixel.
    cy = float(h // 2)
    cx = float(w // 2)
    yy = torch.arange(h, device=device, dtype=torch.float32) - cy
    xx = torch.arange(w, device=device, dtype=torch.float32) - cx
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.sqrt(grid_y ** 2 + grid_x ** 2)


def power_spectrum_2d(x: torch.Tensor) -> torch.Tensor:
    """(C,H,W) or (B,C,H,W) -> |FFT|^2 power spectrum averaged over batch and channel, (H,W)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.float()
    fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    power = (fft.abs() ** 2).mean(dim=(0, 1))  # (H, W)
    return power


def ratio_from_power_spectrum(
    power: torch.Tensor,
    energy_fraction: float = 0.85,
    ratio_min: float = 0.1,
    ratio_max: float = 0.6,
) -> float:
    """Estimate the low-frequency radius ratio from an averaged power spectrum.

    Accumulates energy over radial distance, takes the radius r where
    ``energy_fraction`` of total energy is reached, then ratio = r / (min(H,W)/2).
    """
    if power.dim() != 2:
        raise ValueError(f"power must be (H,W), got {power.shape}")
    h, w = power.shape
    device = power.device
    energy_fraction = float(max(0.5, min(0.99, energy_fraction)))
    r_grid = _radial_distance_grid(h, w, device)
    r_max_bin = int(math.ceil(r_grid.max().item()))
    radial = torch.zeros(r_max_bin + 1, device=device, dtype=power.dtype)
    r_idx = r_grid.clamp(0, r_max_bin).long()
    radial.scatter_add_(0, r_idx.reshape(-1), power.reshape(-1))
    total = radial.sum().clamp_min(1e-12)
    cum = torch.cumsum(radial, dim=0) / total
    hit = torch.nonzero(cum >= energy_fraction, as_tuple=False)
    if hit.numel() == 0:
        r_hit = float(r_max_bin)
    else:
        r_hit = float(hit[0, 0].item())
    half = min(h, w) / 2.0
    ratio = r_hit / max(half, 1e-6)
    return float(max(ratio_min, min(ratio_max, ratio)))


@torch.no_grad()
def estimate_low_freq_ratio_from_tensor(
    x: torch.Tensor,
    energy_fraction: float = 0.85,
) -> float:
    """Estimate the ratio from one (B,C,H,W) batch."""
    return ratio_from_power_spectrum(power_spectrum_2d(x), energy_fraction=energy_fraction)


@torch.no_grad()
def estimate_low_freq_ratio_from_dataset(
    dataset,
    *,
    space: str = "image",
    ae=None,
    scale_factor: float = 1.0,
    device: torch.device | str = "cpu",
    n_samples: int = 512,
    batch_size: int = 16,
    energy_fraction: float = 0.85,
    logger=None,
) -> dict:
    """Estimate ``recon_freq_ratio`` on a training subset by radial energy quantile.

    space:
      - image: protein-channel pixels (B,1,H,W)
      - latent: AE-encoded z * scale_factor (B,4,L,L); requires ``ae``
    """
    from torch.utils.data import DataLoader

    space = (space or "image").lower()
    n_samples = max(1, int(n_samples))
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, n_samples),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    radial_sum = None
    seen = 0
    last_h, last_w = 0, 0
    for batch in loader:
        img = batch[0]
        if space == "image":
            x = img[:, 1:2, :, :].float()
        elif space == "latent":
            if ae is None:
                raise ValueError("latent-space freq ratio estimation requires ae")
            protein = img[:, 1:2, :, :].to(device)
            z = ae.encode(protein).sample().float() * float(scale_factor)
            x = z
        else:
            raise ValueError(f"Unknown space={space}")
        power = power_spectrum_2d(x.cpu())
        h, w = int(power.shape[0]), int(power.shape[1])
        last_h, last_w = h, w
        device_cpu = power.device
        r_grid = _radial_distance_grid(h, w, device_cpu)
        r_max_bin = int(math.ceil(r_grid.max().item()))
        r_idx = r_grid.clamp(0, r_max_bin).long()
        batch_radial = torch.zeros(r_max_bin + 1, dtype=power.dtype)
        batch_radial.scatter_add_(0, r_idx.reshape(-1), power.reshape(-1))
        if radial_sum is None:
            radial_sum = batch_radial
        else:
            if radial_sum.shape[0] < batch_radial.shape[0]:
                pad = batch_radial.shape[0] - radial_sum.shape[0]
                radial_sum = F.pad(radial_sum, (0, pad))
            if batch_radial.shape[0] < radial_sum.shape[0]:
                batch_radial = F.pad(batch_radial, (0, radial_sum.shape[0] - batch_radial.shape[0]))
            radial_sum = radial_sum + batch_radial
        seen += x.shape[0]
        if seen >= n_samples:
            break

    total = radial_sum.clamp_min(1e-12)
    cum = torch.cumsum(total, dim=0) / total.sum()
    hit = torch.nonzero(cum >= energy_fraction, as_tuple=False)
    r_hit = float(hit[0, 0].item()) if hit.numel() else float(radial_sum.shape[0] - 1)
    h, w = last_h, last_w
    half = min(h, w) / 2.0
    ratio = r_hit / max(half, 1e-6)
    ratio = float(max(0.1, min(0.6, ratio)))

    out = {
        "ratio": ratio,
        "energy_fraction": energy_fraction,
        "r_hit_px": r_hit,
        "spatial_hw": (h, w),
        "n_images": seen,
        "space": space,
    }
    if logger is not None:
        logger.info(
            f"[recon-freq-auto] space={space} n={seen} HxW={h}x{w} "
            f"energy_frac={energy_fraction:.2f} → ratio={ratio:.4f} "
            f"(r={r_hit:.1f}px, radius / min(H,W)/2)"
        )
    return out


def create_low_freq_mask(
    h: int,
    w: int,
    ratio: float = 0.3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Circular low-freq mask (H, W bool) of radius ratio * min(H,W)/2 around the fftshifted centre."""
    key = (int(h), int(w), float(ratio))
    if key in _MASK_CACHE:
        m = _MASK_CACHE[key]
        if device is None or m.device == device:
            return m
        return m.to(device)

    # fftshift places DC at h//2; using (h-1)/2 shifts the mask by half a pixel.
    cy = float(h // 2)
    cx = float(w // 2)
    yy = torch.arange(h, device=device, dtype=torch.float32) - cy
    xx = torch.arange(w, device=device, dtype=torch.float32) - cx
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = float(ratio) * min(h, w) / 2.0
    mask = (grid_y ** 2 + grid_x ** 2) <= (radius ** 2)
    _MASK_CACHE[key] = mask
    return mask


def freq_loss_magnitude(
    pred: torch.Tensor,
    target: torch.Tensor,
    low_freq_ratio: float = 0.3,
    low_weight: float = 1.0,
    high_weight: float = 3.0,
) -> torch.Tensor:
    """Weighted low/high-band masked MSE on fftshift(|FFT2|); returns a scalar.

    pred / target: (B, C, H, W), reduced by mean over the batch.
    """
    if pred.shape != target.shape:
        raise ValueError(f"freq_loss shape mismatch: {pred.shape} vs {target.shape}")

    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, dim=(-2, -1)), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))
    pred_mag = pred_fft.abs()
    tgt_mag = target_fft.abs()

    h, w = pred.shape[-2], pred.shape[-1]
    low_mask = create_low_freq_mask(h, w, low_freq_ratio, device=pred.device)
    high_mask = ~low_mask
    # (1, 1, H, W) broadcast to (B, C, H, W)
    low_m = low_mask.view(1, 1, h, w).to(dtype=pred.dtype)
    high_m = high_mask.view(1, 1, h, w).to(dtype=pred.dtype)

    c = pred.shape[1]
    n_low = low_m.sum() * c * pred.shape[0]
    n_high = high_m.sum() * c * pred.shape[0]
    n_low = torch.clamp(n_low, min=1.0)
    n_high = torch.clamp(n_high, min=1.0)

    low_err = F.mse_loss(pred_mag * low_m, tgt_mag * low_m, reduction="sum") / n_low
    high_err = F.mse_loss(pred_mag * high_m, tgt_mag * high_m, reduction="sum") / n_high
    return low_weight * low_err + high_weight * high_err


_LAP_KERNEL_4N = torch.tensor(
    [[0.0, -1.0, 0.0],
     [-1.0, 4.0, -1.0],
     [0.0, -1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)

_LAP_KERNEL_8N = torch.tensor(
    [[-1.0, -1.0, -1.0],
     [-1.0,  8.0, -1.0],
     [-1.0, -1.0, -1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def _laplacian_kernel(neighborhood: int, channels: int, device, dtype) -> torch.Tensor:
    """Depthwise Laplacian kernel (C, 1, 3, 3) for ``F.conv2d(..., groups=C)``."""
    base = _LAP_KERNEL_8N if int(neighborhood) == 8 else _LAP_KERNEL_4N
    k = base.to(device=device, dtype=dtype).expand(channels, 1, 3, 3).contiguous()
    return k


def _laplacian_filter(x: torch.Tensor, neighborhood: int = 4) -> torch.Tensor:
    """Per-channel 3x3 Laplacian on (B,C,H,W) with reflect padding; shape preserved."""
    if x.dim() != 4:
        raise ValueError(f"laplacian expects (B,C,H,W), got {tuple(x.shape)}")
    c = x.shape[1]
    k = _laplacian_kernel(neighborhood, c, x.device, x.dtype)
    x_p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x_p, k, bias=None, stride=1, padding=0, groups=c)


def mean_flat_laplacian_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l1",
    neighborhood: int = 4,
) -> torch.Tensor:
    """Return ``(B,)`` with ``loss = mean_flat(|Lap(pred) - Lap(target)|^p)``.

    loss_type: "l1" -> p=1 (default, more robust to outliers); "l2" -> p=2.
    neighborhood: 4 or 8; 8 includes the diagonals.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"laplacian shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    lap_p = _laplacian_filter(pred, neighborhood=neighborhood)
    lap_t = _laplacian_filter(target, neighborhood=neighborhood)
    diff = lap_p - lap_t
    if str(loss_type).lower() == "l2":
        per = diff.pow(2)
    else:
        per = diff.abs()
    return per.flatten(1).mean(dim=1)


def mean_flat_freq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    low_freq_ratio: float = 0.3,
    low_weight: float = 1.0,
    high_weight: float = 3.0,
) -> torch.Tensor:
    """Per-sample vector (B,), mirroring the train.mean_flat convention."""
    if pred.shape != target.shape:
        raise ValueError(f"freq_loss shape mismatch: {pred.shape} vs {target.shape}")

    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, dim=(-2, -1)), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))
    pred_mag = pred_fft.abs()
    tgt_mag = target_fft.abs()

    h, w = pred.shape[-2], pred.shape[-1]
    low_mask = create_low_freq_mask(h, w, low_freq_ratio, device=pred.device)
    high_mask = ~low_mask
    low_m = low_mask.view(1, 1, h, w).to(dtype=pred.dtype)
    high_m = high_mask.view(1, 1, h, w).to(dtype=pred.dtype)

    c = pred.shape[1]
    n_low = (low_m.sum() * c).clamp_min(1.0)
    n_high = (high_m.sum() * c).clamp_min(1.0)

    low_err = ((pred_mag - tgt_mag) ** 2 * low_m).flatten(1).sum(1) / n_low
    high_err = ((pred_mag - tgt_mag) ** 2 * high_m).flatten(1).sum(1) / n_high
    return low_weight * low_err + high_weight * high_err
