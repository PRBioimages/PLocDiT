#!/usr/bin/env python3
"""Copy + fine-tune HPA VAE on CYNP protein (optional all-ch), then compare vs baseline.

Usage::
  python scripts/finetune_hpa_vae_cynp.py \\
    --data-root /path/to/dataset \\
    --src-ckpt pretrained/hpa_pretrained.bin \\
    --out-dir results_vae_ft/hpa_cynp_ft \\
    --max-steps 4000 --batch-size 16
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def build_vae() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(128, 256, 512),
        latent_channels=4,
        layers_per_block=1,
    )


def load_state_into_vae(vae: AutoencoderKL, ckpt_path: str, trainable: bool) -> None:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        blob = blob["model"]
    elif isinstance(blob, dict) and "module" in blob:
        blob = blob["module"]
    elif isinstance(blob, dict) and "vae" in blob and isinstance(blob["vae"], dict):
        blob = blob["vae"]
    sd = {k[len("vae.") :] if k.startswith("vae.") else k: v for k, v in blob.items()}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "dummy" not in k]
    unexpected = [k for k in unexpected if "dummy" not in k]
    print(f"[vae] load {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    if trainable:
        vae.train()
        for p in vae.parameters():
            p.requires_grad_(True)
    else:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)


def save_vae_bin(vae: AutoencoderKL, path: str, meta: dict | None = None) -> None:
    """Save in CELL-Diff / DiT-subcell compatible flat state_dict form."""
    sd = vae.state_dict()
    # prefix with vae. to match CELL-Diff dumps that use VAEModel
    prefixed = {f"vae.{k}": v.detach().cpu() for k, v in sd.items()}
    payload = {"model": prefixed}
    if meta:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"[save] {path}")


class CypnProteinDataset(Dataset):
    """BGYImg*.png → protein channel (ch1) in [-1,1]."""

    def __init__(self, img_dir: str, img_size: int = 256, channels: str = "protein"):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "BGYImg*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no BGYImg*.png in {img_dir}")
        self.img_size = int(img_size)
        self.channels = channels  # protein | all

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = cv2.imread(self.paths[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"failed to read {self.paths[i]}")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 127.5 - 1.0  # HWC BGR-as-loaded
        # DiT-subcell: ch0=nuc(B), ch1=protein(G), ch2=mt(R)
        chw = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        if self.channels == "all":
            c = int(torch.randint(0, 3, (1,)).item())
            return chw[c : c + 1]
        return chw[1:2]


def laplacian_abs(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1).abs()


def hf_energy(x: torch.Tensor) -> torch.Tensor:
    x01 = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    return laplacian_abs(x01).mean(dim=(1, 2, 3))


@torch.no_grad()
def eval_recon(vae: AutoencoderKL, loader: DataLoader, device: torch.device, max_batches: int = 0):
    vae.eval()
    l1s, mses, ssims, psnrs, hf_ratios = [], [], [], [], []
    n_seen = 0
    for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        x = batch.to(device)
        z = vae.encode(x).latent_dist.mode()
        r = vae.decode(z).sample
        l1s.append((r - x).abs().mean(dim=(1, 2, 3)).cpu())
        mse = ((r - x) ** 2).mean(dim=(1, 2, 3))
        mses.append(mse.cpu())
        x01 = ((x + 1) * 0.5).clamp(0, 1)
        r01 = ((r + 1) * 0.5).clamp(0, 1)
        mse01 = ((r01 - x01) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        psnrs.append((10.0 * torch.log10(1.0 / mse01)).cpu())
        hf_gt = hf_energy(x)
        hf_rec = hf_energy(r)
        hf_ratios.append((hf_rec / (hf_gt + 1e-8)).cpu())
        x_np = x01[:, 0].cpu().numpy()
        r_np = r01[:, 0].cpu().numpy()
        for i in range(x_np.shape[0]):
            ssims.append(
                ssim_metric(x_np[i], r_np[i], data_range=1.0)
            )
        n_seen += x.shape[0]
    def _m(xs):
        t = torch.cat(xs) if xs and torch.is_tensor(xs[0]) else torch.tensor(xs, dtype=torch.float64)
        return float(t.float().mean()), float(t.float().std())
    ssim_arr = np.asarray(ssims, dtype=np.float64)
    return {
        "n": n_seen,
        "l1_mean": _m(l1s)[0],
        "l1_std": _m(l1s)[1],
        "mse_mean": _m(mses)[0],
        "psnr_mean": _m(psnrs)[0],
        "ssim_mean": float(ssim_arr.mean()),
        "ssim_std": float(ssim_arr.std()),
        "hf_ratio_mean": _m(hf_ratios)[0],
        "hf_ratio_std": _m(hf_ratios)[1],
    }


@torch.no_grad()
def save_compare_panel(
    base: AutoencoderKL,
    ft: AutoencoderKL,
    paths: list[str],
    out_png: str,
    device: torch.device,
    img_size: int = 256,
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rows = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy((img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1))[1:2][
            None
        ].to(device)
        zb = base.encode(x).latent_dist.mode()
        rb = base.decode(zb).sample
        zf = ft.encode(x).latent_dist.mode()
        rf = ft.decode(zf).sample

        def to01(t):
            return ((t[0, 0].float().cpu().numpy() + 1) * 0.5).clip(0, 1)

        gt, b, f = to01(x), to01(rb), to01(rf)
        rows.append((os.path.basename(p), gt, b, f, np.abs(b - gt), np.abs(f - gt)))

    if not rows:
        return
    fig = plt.figure(figsize=(12, 1.7 * len(rows)), dpi=200)
    gs = GridSpec(len(rows), 5, figure=fig, wspace=0.04, hspace=0.25, left=0.08, right=0.99, top=0.93, bottom=0.02)
    titles = ["GT protein", "baseline recon", "finetuned recon", "|base-GT|", "|ft-GT|"]
    for r, (name, gt, b, f, eb, ef) in enumerate(rows):
        for c, im in enumerate([gt, b, f, eb, ef]):
            ax = fig.add_subplot(gs[r, c])
            if c >= 3:
                ax.imshow(im, cmap="magma", vmin=0, vmax=max(0.25, float(im.max())))
            else:
                rgb = np.zeros((*im.shape, 3), np.float32)
                rgb[..., 1] = im
                ax.imshow((rgb * 255).astype(np.uint8))
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(name.replace(".png", ""), fontsize=7, rotation=0, labelpad=40, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("HPA VAE: baseline vs CYNP fine-tuned (protein)", fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[panel] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument(
        "--src-ckpt",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "pretrained" / "hpa_pretrained.bin"),
    )
    ap.add_argument("--out-dir", type=str, default="results_vae_ft/hpa_cynp_ft")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--kl-coef", type=float, default=1e-4)
    ap.add_argument("--l1-coef", type=float, default=0.1)
    ap.add_argument("--lap-coef", type=float, default=0.05, help="match high-freq (laplacian)")
    ap.add_argument("--channels", type=str, default="protein", choices=["protein", "all"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-max-batches", type=int, default=40, help="test batches for metrics (~bs*40)")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.src_ckpt)
    if not src.is_file():
        raise FileNotFoundError(src)
    copy_path = out_dir / "hpa_pretrained_COPY.bin"
    if not copy_path.is_file():
        shutil.copy2(src, copy_path)
        print(f"[copy] {src} → {copy_path}")
    else:
        print(f"[copy] exists: {copy_path}")

    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    train_ds = CypnProteinDataset(train_dir, args.img_size, args.channels)
    test_ds = CypnProteinDataset(test_dir, args.img_size, "protein")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    print(f"[data] train={len(train_ds)} test={len(test_ds)} device={device}")

    baseline = build_vae().to(device)
    load_state_into_vae(baseline, str(copy_path), trainable=False)

    print("[eval] baseline (before FT) ...")
    base_metrics = eval_recon(baseline, test_loader, device, max_batches=args.eval_max_batches)
    print(
        f"[baseline] L1={base_metrics['l1_mean']:.4f} SSIM={base_metrics['ssim_mean']:.4f} "
        f"PSNR={base_metrics['psnr_mean']:.2f} HF_ratio={base_metrics['hf_ratio_mean']:.3f} n={base_metrics['n']}"
    )

    vae = build_vae().to(device)
    load_state_into_vae(vae, str(copy_path), trainable=True)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"[train] max_steps={args.max_steps} lr={args.lr} "
        f"kl={args.kl_coef} l1={args.l1_coef} lap={args.lap_coef}"
    )
    vae.train()
    step = 0
    t0 = time.time()
    running = {"mse": 0.0, "l1": 0.0, "kl": 0.0, "lap": 0.0, "n": 0}
    it = iter(train_loader)
    pbar = tqdm(total=args.max_steps, desc="ft")
    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        x = batch.to(device)
        dist = vae.encode(x).latent_dist
        z = dist.sample()
        recon = vae.decode(z).sample
        mse = F.mse_loss(recon, x)
        l1 = (recon - x).abs().mean()
        kl = -0.5 * torch.mean(1 + dist.logvar - dist.mean.pow(2) - dist.logvar.exp())
        lap = F.l1_loss(laplacian_abs(recon), laplacian_abs(x))
        loss = mse + args.l1_coef * l1 + args.kl_coef * kl + args.lap_coef * lap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        step += 1
        running["mse"] += float(mse.detach())
        running["l1"] += float(l1.detach())
        running["kl"] += float(kl.detach())
        running["lap"] += float(lap.detach())
        running["n"] += 1
        pbar.update(1)
        if step % 100 == 0 or step == args.max_steps:
            n = max(1, running["n"])
            pbar.set_postfix(
                mse=f"{running['mse']/n:.4f}",
                l1=f"{running['l1']/n:.4f}",
                hf_lap=f"{running['lap']/n:.4f}",
                kl=f"{running['kl']/n:.4f}",
            )
            running = {"mse": 0.0, "l1": 0.0, "kl": 0.0, "lap": 0.0, "n": 0}
    pbar.close()
    print(f"[train] done in {(time.time()-t0)/60:.1f} min")

    ft_path = out_dir / f"hpa_cynp_ft_steps{args.max_steps}.bin"
    save_vae_bin(
        vae,
        str(ft_path),
        meta={
            "src": str(src),
            "data_root": args.data_root,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "kl_coef": args.kl_coef,
            "l1_coef": args.l1_coef,
            "lap_coef": args.lap_coef,
            "channels": args.channels,
        },
    )

    print("[eval] fine-tuned ...")
    ft_metrics = eval_recon(vae, test_loader, device, max_batches=args.eval_max_batches)
    print(
        f"[finetuned] L1={ft_metrics['l1_mean']:.4f} SSIM={ft_metrics['ssim_mean']:.4f} "
        f"PSNR={ft_metrics['psnr_mean']:.2f} HF_ratio={ft_metrics['hf_ratio_mean']:.3f} n={ft_metrics['n']}"
    )

    csv_path = out_dir / "compare_baseline_vs_ft.csv"
    fieldnames = [
        "model",
        "n",
        "l1_mean",
        "ssim_mean",
        "psnr_mean",
        "hf_ratio_mean",
        "l1_std",
        "ssim_std",
        "hf_ratio_std",
        "mse_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow({"model": "baseline", **base_metrics})
        w.writerow({"model": "finetuned", **ft_metrics})
    print(f"[csv] {csv_path}")

    # visual panel
    rng = np.random.default_rng(args.seed)
    all_test = sorted(glob.glob(os.path.join(test_dir, "BGYImg*.png")))
    pick = [all_test[i] for i in rng.choice(len(all_test), size=min(6, len(all_test)), replace=False)]
    panel = out_dir / "compare_panel_baseline_vs_ft.png"
    save_compare_panel(baseline, vae, pick, str(panel), device, args.img_size)

    def delta(k):
        return ft_metrics[k] - base_metrics[k]

    print("\n======== comparison (ft - baseline; lower L1 is better, higher is better for the rest) ========")
    print(f"  L1       {base_metrics['l1_mean']:.4f} → {ft_metrics['l1_mean']:.4f}  (Δ={delta('l1_mean'):+.4f})")
    print(f"  SSIM     {base_metrics['ssim_mean']:.4f} → {ft_metrics['ssim_mean']:.4f}  (Δ={delta('ssim_mean'):+.4f})")
    print(f"  PSNR     {base_metrics['psnr_mean']:.2f} → {ft_metrics['psnr_mean']:.2f}  (Δ={delta('psnr_mean'):+.2f})")
    print(f"  HF_ratio {base_metrics['hf_ratio_mean']:.3f} → {ft_metrics['hf_ratio_mean']:.3f}  (Δ={delta('hf_ratio_mean'):+.3f})")
    print(f"\n[ft ckpt] {ft_path}")
    print(f"[panel]   {panel}")


if __name__ == "__main__":
    main()
