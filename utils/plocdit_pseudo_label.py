"""Dependency-light pseudo-label helpers for BGY microscopy images.

This module vendors the small, torch-free slice of the CGANCode pseudo-label
pipeline that the evaluation scripts under ``scripts/`` need:

* BGY (blue / green / red) loading with optional intensity pre-processing.
* Parsing of HPA 19-class ``Label`` strings into the five D5 compartments.
* Rule-based compartment scores that act as a model-free pseudo label.
* Constrained normalisation of those scores onto the labelled compartments.

Channel order as read by OpenCV:
  ch0 = Blue  (DAPI / nucleus)
  ch1 = Green (target protein)
  ch2 = Red   (microtubule / structure)

Only numpy and OpenCV are required at runtime.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Five D5 compartments predicted by the pseudo-label pipeline.
D5 = ["Cy", "Np", "Mi", "Nu", "PM"]

# Empirical BGR means of the training split on the [0, 1] scale.
TRAIN_BGR_MEAN = np.array([0.0546, 0.0664, 0.0645], dtype=np.float32)

# HPA 19-class numeric ids (``Label`` column) -> D5 index.
HPA_TO_D5 = {
    0: 1,                              # Nucleoplasm -> Np
    1: 3, 2: 3, 3: 3, 4: 3, 5: 3,      # nuclear substructures -> Nu
    6: 0, 7: 0, 8: 0, 9: 0,            # ER, Golgi, filaments, actin -> Cy
    10: 0, 11: 0, 12: 0,
    13: 4,                             # Plasma membrane -> PM
    14: 2,                             # Mitochondria -> Mi
    15: 0, 16: 0, 17: 0,               # Aggresome, Cytosol, Vesicles -> Cy
    # 18 (Negative) is ignored on purpose.
}

HPA_NAMES = [
    "Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center",
    "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic reticulum", "Golgi Apparatus",
    "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle",
    "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles",
    "Negative",
]


# ---------------------------------------------------------------------------
# BGY loading
# ---------------------------------------------------------------------------

def _read_bgr01(png_path: str, npy_path: str | None = None) -> np.ndarray:
    """Read a cached ``.npy`` when available, otherwise the PNG; return float BGR in [0, 1]."""
    if npy_path and os.path.isfile(npy_path):
        arr = np.load(npy_path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.shape[-1] < 3:
            raise ValueError(f"npy needs >=3 channels: {npy_path} {arr.shape}")
        return arr[..., :3]
    bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(png_path)
    return bgr.astype(np.float32) / 255.0


def percentile_clip_bgr(bgr01: np.ndarray, hi: float = 0.995) -> np.ndarray:
    """Clip each channel at an upper percentile to suppress saturated highlights."""
    out = bgr01.astype(np.float32).copy()
    for c in range(3):
        cap = float(np.quantile(out[:, :, c], hi))
        if cap > 1e-6:
            out[:, :, c] = np.minimum(out[:, :, c], cap)
    return out


def match_train_intensity(bgr01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel affine rescaling towards the training-set BGR means."""
    out = bgr01.astype(np.float32).copy()
    for c in range(3):
        mu = float(out[:, :, c].mean())
        if mu > eps:
            scale = float(TRAIN_BGR_MEAN[c] / mu)
            out[:, :, c] = np.clip(out[:, :, c] * scale, 0.0, 1.0)
    return out


def load_bgy_image(
    png_path: str,
    npy_path: str | None = None,
    img_size: int = 256,
    preprocess: str = "none",
    flip: str | None = None,
) -> np.ndarray:
    """Return a float32 BGR image in [0, 1] with shape (H, W, 3).

    ``preprocess``
        ``none``  resize only.
        ``clip``  99.5% per-channel percentile clip, then resize.
        ``match`` clip plus intensity matching to the training means, then resize.
    ``flip``
        ``None`` | ``h`` | ``v`` | ``hv``.
    """
    bgr = _read_bgr01(png_path, npy_path=npy_path)
    if preprocess == "clip":
        bgr = percentile_clip_bgr(bgr)
    elif preprocess == "match":
        bgr = match_train_intensity(percentile_clip_bgr(bgr))

    if flip:
        if "h" in flip:
            bgr = cv2.flip(bgr, 1)
        if "v" in flip:
            bgr = cv2.flip(bgr, 0)

    if bgr.shape[0] != img_size or bgr.shape[1] != img_size:
        bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return bgr


def normalize_bgy(bgr01: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel standardisation with the given mean and std."""
    img = bgr01.astype(np.float32).copy()
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - float(mean[i])) / float(std[i])
    return img


# ---------------------------------------------------------------------------
# HPA label parsing
# ---------------------------------------------------------------------------

def parse_label_mask(label_str: str) -> tuple[np.ndarray, str]:
    """Parse a ``Label`` string such as ``0|2|16`` into a bool mask over D5."""
    active = np.zeros(5, dtype=bool)
    hpa_ids: list[int] = []
    for part in str(label_str).split("|"):
        part = part.strip()
        if not part:
            continue
        hid = int(part)
        if hid == 18:
            continue
        hpa_ids.append(hid)
        if hid in HPA_TO_D5:
            active[HPA_TO_D5[hid]] = True
    hpa_names = "|".join(HPA_NAMES[h] if h < len(HPA_NAMES) else str(h) for h in hpa_ids)
    d5_names = "|".join(D5[i] for i in range(5) if active[i])
    return active, f"{hpa_names} -> {d5_names}"


def constrained_normalize(raw: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero out inactive compartments and renormalise the rest to sum to one."""
    out = np.where(mask, raw, 0.0)
    s = float(out.sum())
    if s <= eps:
        out = mask.astype(np.float64)
        s = float(out.sum())
    return (out / s).astype(np.float32)


# ---------------------------------------------------------------------------
# Rule-based compartment scores
# ---------------------------------------------------------------------------

def bg_subtract(ch: np.ndarray, q: float = 0.20) -> np.ndarray:
    """Subtract the low quantile of a channel as background."""
    bg = float(np.quantile(ch, q))
    return np.clip(ch - bg, 0.0, None)


def otsu_mask(ch: np.ndarray) -> np.ndarray:
    """Otsu threshold with a quantile fallback, cleaned by an open/close pass."""
    u8 = np.clip(ch * 255.0, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    _thr, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask > 0
    if mask.mean() < 0.01 or mask.mean() > 0.75:
        mask = ch > np.quantile(ch, 0.80)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask


def high_signal_mask(ch: np.ndarray, q: float = 0.85) -> np.ndarray:
    """Keep the brightest ``1 - q`` fraction of the positive pixels."""
    if float(ch.max()) <= 1e-8:
        return np.zeros_like(ch, dtype=bool)
    return ch > np.quantile(ch[ch > 0], q) if np.any(ch > 0) else np.zeros_like(ch, dtype=bool)


def foreground_mask(b: np.ndarray, g: np.ndarray, r: np.ndarray, nucleus: np.ndarray) -> np.ndarray:
    """Union of the nucleus mask and per-channel high-signal regions."""
    masks = [nucleus]
    for ch in (b, g, r):
        sig = bg_subtract(ch)
        if np.any(sig > 0):
            masks.append(sig > np.quantile(sig[sig > 0], 0.65))
    fg = np.logical_or.reduce(masks)
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return fg


def compartment_scores(bgr: np.ndarray) -> np.ndarray:
    """Estimate D5 compartment scores from a BGY image without any model.

    Returns
    -------
    np.ndarray
        Five non-negative scores in ``D5`` order (Cy, Np, Mi, Nu, PM).
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gsig = bg_subtract(g)
    rsig = bg_subtract(r)
    nucleus = otsu_mask(b)
    fg = foreground_mask(b, g, r, nucleus)

    cyto = fg & ~nucleus
    if cyto.sum() < 16:
        cyto = ~nucleus

    # Thin ring just inside the foreground border, used as a PM proxy.
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(fg.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = fg & ~eroded
    if ring.sum() < 16:
        ring = high_signal_mask(gsig, 0.90) & cyto

    # Red channel marks the microtubule / structure stain.
    red_struct = high_signal_mask(rsig, 0.80)
    if red_struct.sum() < 16:
        red_struct = rsig > 0

    # Bright nuclear body distinguishes Nu from the rest of the nucleoplasm.
    nuclear_green = gsig * nucleus
    nuc_body = np.zeros_like(nucleus, dtype=bool)
    if np.any(nuclear_green > 0):
        nuc_body = nucleus & (nuclear_green > np.quantile(nuclear_green[nuclear_green > 0], 0.80))

    scores = np.array([
        float(gsig[cyto].sum()),                  # Cy
        float(gsig[nucleus & ~nuc_body].sum()),   # Np
        float((gsig * red_struct).sum()),         # Mi (MT proxy)
        float(gsig[nuc_body].sum()),              # Nu
        float(gsig[ring].sum()),                  # PM
    ], dtype=np.float64)

    if scores.sum() <= 1e-8:
        scores[:] = 1.0
    return scores


__all__ = [
    "D5",
    "HPA_NAMES",
    "HPA_TO_D5",
    "bg_subtract",
    "compartment_scores",
    "constrained_normalize",
    "foreground_mask",
    "high_signal_mask",
    "load_bgy_image",
    "match_train_intensity",
    "normalize_bgy",
    "otsu_mask",
    "parse_label_mask",
    "percentile_clip_bgr",
]
