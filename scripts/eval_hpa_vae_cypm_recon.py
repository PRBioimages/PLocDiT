"""
Per-channel reconstruction eval of the CELL-Diff HPA pretrained single-channel
VAE on the CYPM three-channel dataset.

Data convention (same as train.py::CYPMDataset):
  cv2.imread on BGYImg*.png gives BGR, so after CHW:
    ch0 = B plane = nucleus
    ch1 = G plane = protein
    ch2 = R plane = microtubule

VAE architecture (inferred from hpa_pretrained.bin):
  diffusers AutoencoderKL, in_channels=1, out_channels=1,
  block_out_channels=(128, 256, 512) (3 down blocks, 2 downsamples),
  latent_channels=4, so a 256 input gives a 64x64x4 latent.

Evaluation flow, per channel of each image:
  1) Normalize to [-1, 1]
  2) encode().latent_dist.mode() -> latent
  3) decode() -> reconstruction
  4) Metrics: MSE@[-1,1], MSE@[0,1], PSNR, SSIM
  Reported as mean ± std per channel; CSV rows are per image and channel.

Usage:
  python scripts/eval_hpa_vae_cypm_recon.py \\
      --data-dir /path/to/dataset \\
      --ckpt pretrained/hpa_pretrained.bin \\
      --max-samples 500 --save-samples 8
"""

import os as _os
import sys as _sys
_SUBDIT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_SUBDIT_ROOT, _os.path.join(_SUBDIT_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from struct_hpa_encoder import DEFAULT_HPA_CKPT  # noqa: E402

CH_NAMES = ["nucleus", "protein", "microtubules"]


def build_vae():
    """Build the AutoencoderKL that matches the hpa_pretrained.bin architecture."""
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(128, 256, 512),
        latent_channels=4,
        layers_per_block=1,
    )


def load_hpa_vae(ckpt_path: str, device: torch.device) -> AutoencoderKL:
    vae = build_vae()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    elif isinstance(ckpt, dict) and "module" in ckpt:
        ckpt = ckpt["module"]

    state_dict = {}
    for k, v in ckpt.items():
        nk = k[len("vae."):] if k.startswith("vae.") else k
        state_dict[nk] = v

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if "dummy" not in k]
    unexpected = [k for k in unexpected if "dummy" not in k]
    print(f"[VAE] loaded  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_image(path: str, img_size: int) -> torch.Tensor | None:
    """Read an image -> (3, H, W) float32 in [-1, 1], channels [nucleus, protein, microtubule]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1)).contiguous()


@torch.no_grad()
def reconstruct(vae: AutoencoderKL, x_1ch: torch.Tensor) -> torch.Tensor:
    """x_1ch: (B,1,H,W) in [-1,1]. Uses mode() to avoid latent sampling noise."""
    latent = vae.encode(x_1ch).latent_dist.mode()
    return vae.decode(latent).sample


def per_sample_metrics(x: torch.Tensor, r: torch.Tensor) -> dict:
    """x, r: (B,1,H,W) in [-1,1]. Returns B-length numpy arrays."""
    x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    r01 = ((r + 1.0) / 2.0).clamp(0.0, 1.0)

    mse = ((x - r) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    mse01 = ((x01 - r01) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    psnr = 10.0 * np.log10(1.0 / np.clip(mse01, 1e-10, None))

    x01_np = x01.cpu().numpy()
    r01_np = r01.cpu().numpy()
    ssim_vals = np.array([
        ssim_metric(x01_np[b, 0], r01_np[b, 0], data_range=1.0)
        for b in range(x01_np.shape[0])
    ])
    return {"mse": mse, "mse01": mse01, "psnr": psnr, "ssim": ssim_vals}


def save_visualization(
    save_path: str,
    orig: torch.Tensor,        # (N, 3, H, W) in [-1,1], channels=[nucleus, protein, microtubule]
    recon: torch.Tensor,       # (N, 3, H, W) in [-1,1]
    n_samples: int,
):
    n = min(n_samples, orig.size(0))
    if n == 0:
        return
    H = orig.size(-2)
    W = orig.size(-1)
    pad = 2
    # One row per sample: orig0,recon0 | orig1,recon1 | orig2,recon2 (6 columns)
    cols = 6
    rows = n
    canvas = np.zeros((rows * (H + pad), cols * (W + pad), 3), dtype=np.uint8)

    def to_u8(t):
        """(H,W) in [-1,1] -> (H,W,3) uint8 with the grayscale plane cloned to 3 channels."""
        arr = ((t.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
        arr = (arr * 255).astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)

    for i in range(n):
        for ch in range(3):
            o = to_u8(orig[i, ch])
            r = to_u8(recon[i, ch])
            # Tint by channel: nucleus=blue, protein=green, microtubule=red
            colormask = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)}[ch]  # BGR tuple 0/1
            o_c = (o.astype(np.float32) * np.array(colormask, dtype=np.float32)).clip(0, 255).astype(np.uint8)
            r_c = (r.astype(np.float32) * np.array(colormask, dtype=np.float32)).clip(0, 255).astype(np.uint8)
            y = i * (H + pad)
            x_o = (ch * 2) * (W + pad)
            x_r = (ch * 2 + 1) * (W + pad)
            canvas[y:y + H, x_o:x_o + W] = o_c
            canvas[y:y + H, x_r:x_r + W] = r_c

    # Header text
    header = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "orig|recon : nucleus(B)  protein(G)  microtubules(R)",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.concatenate([header, canvas], axis=0)

    cv2.imwrite(save_path, canvas)
    print(f"[vis] saved {save_path}  shape={canvas.shape}")


def main():
    parser = argparse.ArgumentParser(description="CELL-Diff HPA VAE on CYPM reconstruction eval")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--ckpt", default=DEFAULT_HPA_CKPT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="0 means use all")
    parser.add_argument("--save-samples", type=int, default=8,
                        help="Number of samples in the visualization panel; 0 disables it")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Run inference in half precision (faster, slightly worse metrics)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    vae = load_hpa_vae(args.ckpt, device)
    if args.fp16 and device.type == "cuda":
        vae = vae.half()

    # Collect files
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.png"), recursive=True))
    if args.max_samples and args.max_samples > 0:
        files = files[:args.max_samples]
    print(f"[data] {len(files)} images from {args.data_dir}")
    if not files:
        raise SystemExit("no .png found")

    per_ch: dict[str, dict[str, list]] = {
        ch: {"mse": [], "mse01": [], "psnr": [], "ssim": []} for ch in CH_NAMES
    }
    per_sample_rows: list[dict] = []
    # Cache the first save_samples images for visualization
    vis_orig, vis_recon = [], []

    for i in tqdm(range(0, len(files), args.batch_size), desc="eval"):
        batch_files = files[i:i + args.batch_size]
        imgs = []
        kept = []
        for f in batch_files:
            t = load_image(f, args.img_size)
            if t is not None:
                imgs.append(t)
                kept.append(f)
        if not imgs:
            continue
        img_batch = torch.stack(imgs, dim=0).to(device)  # (B,3,H,W) in [-1,1]
        if args.fp16 and device.type == "cuda":
            img_batch = img_batch.half()

        recon_full = torch.zeros_like(img_batch)
        batch_metrics = {ch: {} for ch in CH_NAMES}

        for ch_idx, ch_name in enumerate(CH_NAMES):
            x = img_batch[:, ch_idx:ch_idx + 1]
            r = reconstruct(vae, x)
            recon_full[:, ch_idx:ch_idx + 1] = r
            m = per_sample_metrics(x.float(), r.float())
            batch_metrics[ch_name] = m
            for k, v in m.items():
                per_ch[ch_name][k].extend(v.tolist())

        # One CSV row per sample
        for bi, fpath in enumerate(kept):
            row = {"file": os.path.basename(fpath)}
            for ch in CH_NAMES:
                for k in ["mse", "mse01", "psnr", "ssim"]:
                    row[f"{ch}_{k}"] = float(batch_metrics[ch][k][bi])
            per_sample_rows.append(row)

        # Visualization cache
        need = max(0, args.save_samples - sum(x.size(0) for x in vis_orig))
        if need > 0:
            vis_orig.append(img_batch[:need].float().cpu())
            vis_recon.append(recon_full[:need].float().cpu())

    # ==== Summary ====
    print("\n=== Per-channel reconstruction metrics (mean ± std) ===")
    hdr = f"{'channel':<14} {'MSE[-1,1]':<20} {'MSE[0,1]':<20} {'PSNR(dB)':<18} {'SSIM':<16}  N"
    print(hdr)
    print("-" * len(hdr))
    summary_rows = []
    for ch in CH_NAMES:
        d = per_ch[ch]
        n = len(d["mse"])
        stats = {}
        for k in ["mse", "mse01", "psnr", "ssim"]:
            arr = np.array(d[k], dtype=np.float64)
            stats[f"{k}_mean"] = float(arr.mean()) if n else float("nan")
            stats[f"{k}_std"] = float(arr.std()) if n else float("nan")
        summary_rows.append({"channel": ch, "n": n, **stats})
        print(
            f"{ch:<14} "
            f"{stats['mse_mean']:.5f}±{stats['mse_std']:.5f}   "
            f"{stats['mse01_mean']:.5f}±{stats['mse01_std']:.5f}   "
            f"{stats['psnr_mean']:.3f}±{stats['psnr_std']:.3f}        "
            f"{stats['ssim_mean']:.4f}±{stats['ssim_std']:.4f}   "
            f"{n}"
        )

    # ==== Write CSV ====
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[csv] summary → {summary_csv}")

    per_sample_csv = os.path.join(args.out_dir, "per_sample.csv")
    with open(per_sample_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        w.writeheader()
        w.writerows(per_sample_rows)
    print(f"[csv] per-sample → {per_sample_csv}")

    # ==== Save visualization ====
    if args.save_samples > 0 and vis_orig:
        vo = torch.cat(vis_orig, dim=0)[:args.save_samples]
        vr = torch.cat(vis_recon, dim=0)[:args.save_samples]
        save_visualization(os.path.join(args.out_dir, "recon_vis.png"), vo, vr, args.save_samples)


if __name__ == "__main__":
    main()
