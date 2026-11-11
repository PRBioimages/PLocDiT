"""Evaluate a DiT-main checkpoint on a fixed test subset (SSIM/PSNR/FID + visualizations).

Same subset protocol as cfg_sweep: the first column of --subset-indices-file holds
positions in the sorted test glob.
--label-source oracle: testlabels h5; pseudo: test_pseudo_labels.csv (idx=position).
--cfg-freq-weighted: enable FA-CFG.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_SUBDIT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_SUBDIT_ROOT, _os.path.join(_SUBDIT_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse
import csv
import importlib.util
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DIT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _DIT_ROOT not in sys.path:
    sys.path.insert(0, _DIT_ROOT)

_spec = importlib.util.spec_from_file_location(
    "_eval_dit_metrics",
    os.path.join(_SCRIPT_DIR, "eval_dit_metrics.py"),
)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

load_everything = _eval.load_everything
_load_val_samples = _eval._load_val_samples
_batch_images_3ch = _eval._batch_images_3ch
_FIDAccumulator = _eval._FIDAccumulator
decode_protein = _eval.decode_protein
compute_ssim_psnr = _eval.compute_ssim_psnr
create_diffusion = _eval.create_diffusion
_protein_tensor_to_u8_png = _eval._protein_tensor_to_u8_png

from models import prepare_cfg_ddim_sampling, take_cfg_protein_latent  # noqa: E402


def _nuc_mask_from_ch0(nuc_2d: np.ndarray, cell_thr: float = 0.05, nuc_percentile: float = 60.0) -> np.ndarray:
    cell = np.abs(nuc_2d) > cell_thr
    if not cell.any():
        return np.zeros_like(nuc_2d, dtype=bool)
    thr = float(np.percentile(nuc_2d[cell], nuc_percentile))
    thr = max(thr, 0.04)
    return (nuc_2d >= thr) & cell


def clean_cyto_struct_with_nuc_mask(
    img_bchw: torch.Tensor,
    *,
    nuc_ch: int,
    cyto_ch: int,
    bg_value: float = -1.0,
    cell_thr: float = 0.05,
    nuc_percentile: float = 60.0,
    dilate_px: int = 0,
) -> torch.Tensor:
    """Set ch2 inside the nucleus mask to background -1; returns the cleaned (B,3,H,W)."""
    out = img_bchw.clone()
    for i in range(out.shape[0]):
        nuc_np = out[i, nuc_ch].detach().cpu().numpy()
        m = _nuc_mask_from_ch0(nuc_np, cell_thr=cell_thr, nuc_percentile=nuc_percentile)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            m = cv2.dilate(m.astype(np.uint8), k, iterations=1).astype(bool)
        out[i, cyto_ch][torch.from_numpy(m).to(out.device)] = bg_value
    return out


def _load_pseudo_label_csv(csv_path: str) -> tuple[dict[int, np.ndarray], set[int]]:
    label_map: dict[int, np.ndarray] = {}
    excluded: set[int] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row.get("index") or row["idx"])
            if int(float(row.get("use_pseudo", 1))) == 0:
                excluded.add(idx)
                continue
            cyto = float(row.get("p_cyt_pseudo") or row.get("p_cyto"))
            nuc = float(row.get("p_nuc_pseudo") or row.get("p_nucleus"))
            label_map[idx] = np.array([cyto, nuc], dtype=np.float32)
    return label_map, excluded


def _load_subset_positions(path: str) -> np.ndarray:
    positions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            positions.append(int(line.split()[0]))
    return np.array(sorted(positions), dtype=np.int64)


def _load_subset(args, train_args, device):
    dr = os.path.normpath(str(args.data_root).strip())
    tag = (args.dataset_h5_tag or "CyNp").strip()
    val_dir = (args.val_dir or "").strip() or os.path.join(dr, "test")
    val_h5 = os.path.join(dr, f"testlabels_{tag}.h5")
    print(f"[data] val_dir={val_dir}  labels={val_h5}")
    all_samples = _load_val_samples(val_dir, val_h5, 0, train_args.struct_ch_indices, device)
    positions = _load_subset_positions(args.subset_indices_file)
    subset = [all_samples[i] for i in positions]
    file_idxs = [s[0] for s in subset]
    print(f"[subset] n={len(positions)} from {args.subset_indices_file}")
    print(f"[subset] positions[:5]={positions[:5].tolist()}  BGYImg[:5]={file_idxs[:5]}")
    return subset, positions, file_idxs


def _precompute_noise_cache(n, gen_batch, main_L, device, seed):
    cache = {}
    for start in range(0, n, gen_batch):
        b = min(gen_batch, n - start)
        g = torch.Generator(device=device).manual_seed(int(seed) + start)
        cache[start] = torch.randn(b, 4, main_L, main_L, device=device, generator=g)
    return cache


def _save_vis_first(real, fake, out_dir, n=16):
    if n <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, real.shape[0])
    for i in range(n):
        r = _protein_tensor_to_u8_png(real[i])
        f = _protein_tensor_to_u8_png(fake[i])
        combined = np.concatenate([r, f], axis=0)
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}_real_fake.png"), combined)
    h, w = real.shape[-2], real.shape[-1]
    strip = np.zeros((2 * h, n * w), dtype=np.uint8)
    for i in range(n):
        strip[0:h, i * w : (i + 1) * w] = _protein_tensor_to_u8_png(real[i])
        strip[h : 2 * h, i * w : (i + 1) * w] = _protein_tensor_to_u8_png(fake[i])
    cv2.imwrite(os.path.join(out_dir, "combined_first16.png"), strip)
    print(f"[vis] first {n} → {out_dir}/combined_first16.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--dataset-h5-tag", type=str, default="CyNp")
    parser.add_argument("--subset-indices-file", type=str, required=True)
    parser.add_argument("--val-dir", type=str, default="")
    parser.add_argument(
        "--label-source",
        type=str,
        default="oracle",
        choices=["oracle", "pseudo"],
        help="oracle=testlabels h5; pseudo=--pseudo-csv (idx=subset position)",
    )
    parser.add_argument(
        "--pseudo-csv",
        type=str,
        default="",
    )
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument(
        "--flow-fast",
        action="store_true",
        help="Disable Heun for flow sampling (Euler), matching endpoint-interp pseudo labels/eval",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.25)
    parser.add_argument("--cfg-freq-weighted", action="store_true")
    parser.add_argument("--cfg-w-low", type=float, default=None)
    parser.add_argument("--cfg-w-high", type=float, default=1.0)
    parser.add_argument("--cfg-freq-mode", type=str, default="fft", choices=["fft", "bilinear"])
    parser.add_argument("--cfg-freq-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument(
        "--clean-cyto-struct",
        action="store_true",
        help="Clean the ch2 cytoplasm/microtubule struct with the ch0 nucleus mask before encoding",
    )
    parser.add_argument("--nuc-percentile", type=float, default=60.0)
    parser.add_argument("--dilate-px", type=int, default=1)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--vis-first", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[device] {device}")
    label_source = str(args.label_source).lower()
    label_map, excluded = {}, set()
    if label_source == "pseudo":
        label_map, excluded = _load_pseudo_label_csv(args.pseudo_csv)
        print(f"[labels] pseudo CSV: {args.pseudo_csv}  entries={len(label_map)}  excluded={len(excluded)}")
    else:
        print("[labels] oracle from testlabels h5")
    if args.cfg_freq_weighted:
        wl = float(args.cfg_w_low if args.cfg_w_low is not None else args.cfg_scale)
        print(f"[facfg] wl={wl} wh={args.cfg_w_high} ratio={args.cfg_freq_ratio} mode={args.cfg_freq_mode}")

    model, struct_encoder, ae, scale_factor, struct_scale_factor, main_L, train_args = (
        load_everything(args.ckpt, device)
    )

    diff_type = getattr(train_args, "diffusion_type", "ddpm")
    flow_heun = False if args.flow_fast else bool(getattr(train_args, "flow_use_heun", False))
    diffusion = create_diffusion(
        timestep_respacing=str(args.ddim_steps),
        learn_sigma=False,
        predict_xstart=str(getattr(train_args, "dit_parameterization", "eps")).lower() == "x0",
        diffusion_type=diff_type,
        flow_num_steps=args.ddim_steps,
        flow_sigma_min=float(getattr(train_args, "flow_sigma_min", 0.0)),
        flow_use_heun=flow_heun,
        flow_time_shift_mu=float(getattr(train_args, "flow_time_shift_mu", 0.0)),
    )
    x0_clamp = float(getattr(train_args, "ddim_x0_clamp", 4.0) or 0.0)
    denoised_fn = (lambda x, c=x0_clamp: x.clamp(-c, c)) if x0_clamp > 0 else None
    print(
        f"[diffusion] type={diff_type} steps={args.ddim_steps} heun={flow_heun} "
        f"cfg={args.cfg_scale} seed={args.seed} x0_clamp={x0_clamp}"
    )
    if args.clean_cyto_struct:
        print(
            f"[struct-clean] cyto ch mask from nuc ch | "
            f"nuc_percentile={args.nuc_percentile} dilate_px={args.dilate_px}"
        )
    print("[gen-mode] mlp_direct (oracle label → LabelEmbedder MLP → sample)")

    subset, positions, file_idxs = _load_subset(args, train_args, device)
    n = len(subset)
    gen_bs = max(1, int(args.gen_batch_size))
    noise_cache = _precompute_noise_cache(n, gen_bs, main_L, device, args.seed)

    oracle_t = torch.stack([subset[i][2] for i in range(n)], dim=0)
    if label_source == "pseudo":
        pseudo_labels = []
        for pos in positions.tolist():
            if pos in excluded:
                raise ValueError(f"position={pos} use_pseudo=0")
            if pos not in label_map:
                raise KeyError(f"position={pos} missing from the pseudo-label CSV")
            pseudo_labels.append(torch.from_numpy(label_map[pos].astype(np.float32)))
        label_t = torch.stack(pseudo_labels, dim=0)
        print(f"[labels] first 3 pseudo y: {label_t[:3].tolist()}")
        print(f"[labels] first 3 oracle y: {label_t[:3].tolist()}")
    else:
        label_t = oracle_t
        print(f"[labels] first 3 oracle y: {label_t[:3].tolist()}")

    fid = _FIDAccumulator(feature_dim=2048, device=device)
    ssim_all, psnr_all = [], []
    per_rows = []
    fakes_cpu = []
    reals_cpu = []

    print("[fid] real ...")
    for start in tqdm(range(0, n, gen_bs), desc="real"):
        batch = subset[start : start + gen_bs]
        real = _batch_images_3ch(batch).to(device)[:, 1:2]
        fid.update(real, real=True)

    t0 = time.time()
    gen_desc = "gen_pseudo" if label_source == "pseudo" else "gen_oracle"
    for start in tqdm(range(0, n, gen_bs), desc=gen_desc):
        end = start + gen_bs
        batch = subset[start:end]
        b = len(batch)
        img_3ch = _batch_images_3ch(batch).to(device)
        labels = label_t[start:end].to(device)
        real = img_3ch[:, 1:2]
        c0, c1 = int(train_args.struct_ch_indices[0]), int(train_args.struct_ch_indices[1])
        struct_in = img_3ch
        if args.clean_cyto_struct:
            struct_in = clean_cyto_struct_with_nuc_mask(
                img_3ch,
                nuc_ch=c0,
                cyto_ch=c1,
                nuc_percentile=args.nuc_percentile,
                dilate_px=args.dilate_px,
            )
        z_struct = struct_encoder(struct_in[:, [c0, c1]]) * struct_scale_factor
        noise = noise_cache[start]

        sample_shape, noise_in, mk, loop_model, use_cfg = prepare_cfg_ddim_sampling(
            b, noise, z_struct, labels, float(args.cfg_scale), diff_type, model,
        )
        if args.cfg_freq_weighted:
            mk["cfg_freq_weighted"] = True
            mk["cfg_w_low"] = float(args.cfg_w_low if args.cfg_w_low is not None else args.cfg_scale)
            mk["cfg_w_high"] = float(args.cfg_w_high)
            mk["cfg_freq_mode"] = str(args.cfg_freq_mode)
            mk["cfg_freq_ratio"] = float(args.cfg_freq_ratio)
            mk["cfg_freq_time_schedule"] = True
        samples = diffusion.ddim_sample_loop(
            loop_model,
            shape=sample_shape,
            noise=noise_in,
            clip_denoised=False,
            denoised_fn=denoised_fn,
            model_kwargs=mk,
            progress=False,
            device=device,
            eta=args.ddim_eta,
        )
        z_out = take_cfg_protein_latent(samples, b, use_cfg, diff_type)
        fake = decode_protein(ae, z_out[:, :4], scale_factor)
        fid.update(fake, real=False)
        ssim_b, psnr_b = compute_ssim_psnr(real, fake)
        ssim_all.extend(ssim_b.tolist())
        psnr_all.extend(psnr_b.tolist())
        fakes_cpu.append(fake.cpu())
        reals_cpu.append(real.cpu())
        for j in range(b):
            pos = int(positions[start + j])
            per_rows.append({
                "position": pos,
                "file_idx": file_idxs[start + j],
                "p_cyt_cond": float(labels[j, 0].item()),
                "p_nuc_cond": float(labels[j, 1].item()),
                "p_cyt_oracle": float(oracle_t[start + j, 0].item()),
                "p_nuc_oracle": float(oracle_t[start + j, 1].item()),
                "ssim": float(ssim_b[j]),
                "psnr": float(psnr_b[j]),
            })

    elapsed = time.time() - t0
    fid_score = fid.compute()
    ssim_arr = np.array(ssim_all)
    psnr_arr = np.array(psnr_all)

    n_endpoint = int((((label_t[:, 0] == 0) | (label_t[:, 1] == 0)).sum()).item())
    n_partial = n - n_endpoint

    summary = {
        "ckpt": os.path.basename(args.ckpt),
        "n_subset": n,
        "n_endpoint": n_endpoint,
        "n_partial": n_partial,
        "method": "mlp_direct_oracle",
        "ddim_steps": args.ddim_steps,
        "flow_fast": int(args.flow_fast),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "label_source": label_source,
        "clean_cyto_struct": int(args.clean_cyto_struct),
        "nuc_percentile": args.nuc_percentile if args.clean_cyto_struct else "",
        "dilate_px": args.dilate_px if args.clean_cyto_struct else "",
        "pseudo_csv": (args.pseudo_csv or "").strip() if label_source == "pseudo" else "",
        "cfg_freq_weighted": bool(args.cfg_freq_weighted),
        "cfg_w_low": float(args.cfg_w_low if args.cfg_w_low is not None else args.cfg_scale),
        "cfg_w_high": float(args.cfg_w_high),
        "cfg_freq_ratio": float(args.cfg_freq_ratio),
        "val_dir": (args.val_dir or "").strip() or os.path.join(
            os.path.normpath(str(args.data_root).strip()), "test"
        ),
        "ssim_mean": float(ssim_arr.mean()),
        "ssim_std": float(ssim_arr.std()),
        "psnr_mean": float(psnr_arr.mean()),
        "psnr_std": float(psnr_arr.std()),
        "fid": float(fid_score),
        "elapsed_min": elapsed / 60.0,
    }
    print(
        f"\n[result] SSIM={summary['ssim_mean']:.4f}±{summary['ssim_std']:.4f}  "
        f"PSNR={summary['psnr_mean']:.2f}  FID={summary['fid']:.2f}  "
        f"time={elapsed/60:.1f}min"
    )

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    with open(os.path.join(args.out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    if int(args.vis_first) > 0:
        fake_all = torch.cat(fakes_cpu, dim=0)
        real_all = torch.cat(reals_cpu, dim=0)
        _save_vis_first(real_all, fake_all, os.path.join(args.out_dir, "vis_first"), n=int(args.vis_first))

    with open(os.path.join(args.out_dir, "run_meta.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}={v}\n")
        f.write(f"subset_indices_file={args.subset_indices_file}\n")
    print(f"[done] → {args.out_dir}")


if __name__ == "__main__":
    main()
