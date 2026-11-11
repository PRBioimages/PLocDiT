#!/usr/bin/env python3
"""Compare plain sampling against DFA-CFG (Dynamic Frequency-Aware CFG) sampling.

For one checkpoint and the same latent noise, this script renders a side-by-side
row per sample:

    [nucleus (struct) | microtubule (struct) | generated (no CFG) |
     generated (DFA-CFG) | ground-truth protein]

DFA-CFG keeps the low-frequency part of the CFG update close to the plain
``cfg_scale`` and damps the high-frequency part, scheduling both band width and
weights over diffusion time. Defaults follow DiT-main:
``fft`` low-pass, ``freq_ratio=0.3``, ``schedule=dual_linear``.

Example:
    python sample_cfg.py \
        --dit-ckpt results/dit_x0_adv_cynp_v2/002-DiT-B-4/checkpoints/0010000.pt \
        --img-dir /path/to/CYNP_ALL/test \
        --labels-h5 /path/to/testlabels_CyNp.h5 \
        --cfg-scale 1.25 \
        --dfa-w-low 1.25 --dfa-w-high 1.0 \
        --dfa-schedule dual_linear --dfa-ratio 0.3 \
        --n-samples 8 --ddim-steps 250 --out-dir ./cfg_compare

Disable the frequency weighting (plain CFG):
    python sample_cfg.py ... --no-dfa-cfg
"""

import os as _os
import sys as _sys
_PLOCDIT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_PLOCDIT_ROOT, _os.path.join(_PLOCDIT_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import os
import random

import cv2
import h5py
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from dit_model_build import build_dit_seq_from_args
from models import CFGModelWrapper
from diffusion import create_diffusion
from struct_hpa_encoder import (
    DEFAULT_HPA_CKPT,
    load_hpa_main_ae,
    load_hpa_struct_encoder,
)
from utils.dynamic_freq_sampling import (
    DFACFGConfig,
    DFACFGWrapper,
    build_dfa_cfg_kwargs,
)

IMG_SIZE = 256


# ---------------------------------------------------------------------------
# Small image helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Read a BGR uint8 image and resize it to ``IMG_SIZE``."""
    img = cv2.imread(path)
    assert img is not None, "failed to read image: %s" % path
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def img_to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 -> (1, 3, H, W) float32 in [-1, 1]."""
    t = bgr.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).to(device)


def tensor_to_u8(t: torch.Tensor) -> np.ndarray:
    """(1, 1, H, W) or (1, H, W) float32 in [-1, 1] -> (H, W) uint8."""
    x = t.squeeze().float().cpu().numpy()
    x = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def colorize_gray(arr: np.ndarray, color: tuple) -> np.ndarray:
    """(H, W) uint8 -> (H, W, 3) BGR tinted with ``color``."""
    f = arr.astype(np.float32) / 255.0
    out = np.stack([f * color[0], f * color[1], f * color[2]], axis=-1)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def make_label_row(img_list: list, titles: list, h: int = 24) -> np.ndarray:
    """Render a title strip above a horizontal image row."""
    total_w = sum(x.shape[1] for x in img_list)
    header = np.zeros((h, total_w, 3), dtype=np.uint8)
    x_off = 0
    for im, title in zip(img_list, titles):
        cv2.putText(
            header, title,
            (x_off + 4, h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
        )
        x_off += im.shape[1]
    return header


# ---------------------------------------------------------------------------
# Sampling core
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one(
    model,
    diffusion,
    ae,
    struct_encoder,
    img_tensor: torch.Tensor,       # (1, 3, H, W) in [-1, 1]
    label: torch.Tensor,            # (1, 2)
    scale_factor: float,
    struct_scale_factor: float,
    struct_ch_indices: tuple,
    cfg_scale: float,
    latent_size: int,
    device: torch.device,
    dfa_cfg: DFACFGConfig,
    is_flow: bool,
    ddim_eta: float = 0.0,
    seed: int = 0,
):
    """Return ``(gen_nocfg_u8, gen_dfa_u8)``, both (H, W) uint8.

    Both branches share the same initial noise so the only difference is the
    guidance rule.
    """
    torch.manual_seed(seed)

    # Structure latent from the nucleus / microtubule channels.
    ch0, ch1 = struct_ch_indices
    struct_2ch = img_tensor[:, [ch0, ch1]].contiguous()      # (1, 2, H, W)
    z_struct = struct_encoder(struct_2ch) * struct_scale_factor  # (1, 8, L, L)

    # --- Branch 1: no guidance ---
    z_noise = torch.randn(1, 4, latent_size, latent_size, device=device)
    samples_nocfg = diffusion.ddim_sample_loop(
        model,
        shape=(1, 4, latent_size, latent_size),
        noise=z_noise.clone(),
        clip_denoised=False,
        model_kwargs=dict(z_struct=z_struct, y=label),
        progress=False,
        device=device,
        eta=ddim_eta,
    )
    # x0 parameterization: clamp the latent before decoding.
    z_nocfg = samples_nocfg[:, :4].clamp(-6.0, 6.0)
    _dec = ae.decode(z_nocfg / scale_factor)
    gen_nocfg = _dec.sample if hasattr(_dec, "sample") else _dec
    gen_nocfg_u8 = tensor_to_u8(gen_nocfg)

    # --- Branch 2: guided (plain CFG vs. DFA-CFG) ---
    # Only conditional inputs are passed; the sampler expands the batch by
    # appending zero (null) conditioning for the unconditional half.
    noise_2 = torch.cat([z_noise, z_noise], dim=0)            # (2, 4, L, L)
    model_kwargs = dict(z_struct=z_struct, y=label, cfg_scale=cfg_scale)
    if is_flow:
        # FlowMatching performs the frequency-aware mixing inside the loop.
        model_kwargs.update(build_dfa_cfg_kwargs(dfa_cfg, cfg_scale))
        loop_model = model
    elif dfa_cfg.enabled:
        # DDPM / x0 with DFA-CFG: mix on the denoiser output.
        loop_model = DFACFGWrapper(model, cfg_scale, dfa_cfg)
    else:
        # DDPM / x0 with --no-dfa-cfg: plain CFG on the denoiser output.
        loop_model = CFGModelWrapper(model, cfg_scale)

    samples_cfg = diffusion.ddim_sample_loop(
        loop_model,
        shape=(2, 4, latent_size, latent_size),
        noise=noise_2,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=False,
        device=device,
        eta=ddim_eta,
    )
    z_cfg = samples_cfg[:1, :4].clamp(-6.0, 6.0)
    _dec = ae.decode(z_cfg / scale_factor)
    gen_cfg = _dec.sample if hasattr(_dec, "sample") else _dec
    gen_cfg_u8 = tensor_to_u8(gen_cfg)

    return gen_nocfg_u8, gen_cfg_u8


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device] %s" % device)

    # --- Checkpoint ---
    ckpt = torch.load(args.dit_ckpt, map_location=device, weights_only=False)
    train_args = ckpt.get("args")
    if train_args is None:
        raise ValueError("checkpoint has no 'args'; cannot rebuild the model")

    scale_factor = float(ckpt.get("scale_factor", 1.2731))
    struct_scale_factor = float(ckpt.get("struct_scale_factor", 1.3810))
    print("[scale] scale_factor=%.6f  struct_scale_factor=%.6f"
          % (scale_factor, struct_scale_factor))

    # --- Model ---
    model, latent_size, _ = build_dit_seq_from_args(train_args, device)
    state_dict = ckpt["model"]
    # Fill keys that older checkpoints may lack with freshly initialized values.
    for key in model.state_dict():
        if key not in state_dict:
            print("[ckpt] missing %s, using random init" % key)
            state_dict[key] = model.state_dict()[key].clone()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[model] loaded, %.3fM parameters" % (sum(p.numel() for p in model.parameters()) / 1e6))

    # --- Diffusion type / parameterization ---
    pred_type = getattr(train_args, "dit_parameterization", "eps")
    diff_type = str(getattr(train_args, "diffusion_type", "ddpm"))
    is_flow = diff_type == "flow"
    predict_xstart = pred_type == "x0"

    dfa_cfg = DFACFGConfig(
        enabled=not args.no_dfa_cfg,
        freq_mode=args.dfa_mode,
        freq_ratio=args.dfa_ratio,
        freq_ratio_min=args.dfa_ratio_min,
        freq_schedule=args.dfa_schedule,
        time_schedule=args.dfa_time_schedule,
        w_low=args.dfa_w_low,
        w_high=args.dfa_w_high,
        w_low_min=args.dfa_w_low_min,
        w_high_min=args.dfa_w_high_min,
    )
    print("[cfg] prediction_type=%s  diffusion_type=%s  cfg_scale=%s"
          % (pred_type, diff_type, args.cfg_scale))
    print("[dfa-cfg] %s" % dfa_cfg.describe(args.cfg_scale))

    # --- Autoencoders ---
    ae_ckpt = os.path.abspath(os.path.expanduser((args.main_ae_ckpt or "").strip() or DEFAULT_HPA_CKPT))
    ae = load_hpa_main_ae(ae_ckpt, device)
    print("[ae] HPA VAE loaded from %s" % ae_ckpt)

    hpa_vae_ckpt = os.path.abspath(os.path.expanduser((args.hpa_vae_ckpt or "").strip() or DEFAULT_HPA_CKPT))
    struct_encoder = load_hpa_struct_encoder(
        ckpt_path=hpa_vae_ckpt,
        device=device,
        latent_size=latent_size,
    )
    print("[struct] HPA VAE struct encoder loaded, latent_size=%d" % latent_size)

    # --- Sampler ---
    if is_flow:
        diffusion = create_diffusion(
            str(args.ddim_steps),
            learn_sigma=False,
            diffusion_type="flow",
            flow_num_steps=args.ddim_steps,
            flow_sigma_min=float(getattr(train_args, "flow_sigma_min", 0.0)),
            flow_use_heun=bool(getattr(train_args, "flow_use_heun", False)),
            flow_time_shift_mu=float(getattr(train_args, "flow_time_shift_mu", 0.0)),
        )
    else:
        diffusion = create_diffusion(
            str(args.ddim_steps),
            learn_sigma=False,
            predict_xstart=predict_xstart,
        )
    print("[diffusion] type=%s  DDIM steps=%d  predict_xstart=%s"
          % (diff_type, args.ddim_steps, predict_xstart))

    # --- Labels ---
    labels = None
    if args.labels_h5 and os.path.isfile(args.labels_h5):
        with h5py.File(args.labels_h5, "r") as f:
            labels = f["labels"][()]
        print("[label] %d records from %s" % (len(labels), args.labels_h5))

    # --- Sample indices ---
    import glob as _glob
    all_imgs = sorted(_glob.glob(os.path.join(args.img_dir, "BGYImg*.png")))
    if not all_imgs:
        raise FileNotFoundError("no BGYImg*.png under %s" % args.img_dir)
    print("[data] %d images" % len(all_imgs))

    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
    else:
        random.seed(args.seed)
        indices = sorted(random.sample(range(len(all_imgs)), min(args.n_samples, len(all_imgs))))
    print("[data] sampling %d indices: %s" % (len(indices), indices))

    os.makedirs(args.out_dir, exist_ok=True)

    struct_ch_indices = tuple(getattr(train_args, "struct_ch_indices", [0, 2])[:2])
    cfg_title = "Generated(cfg=%s)" % args.cfg_scale
    if dfa_cfg.enabled:
        cfg_title = "Generated(cfg=%s, DFA-CFG %s)" % (args.cfg_scale, args.dfa_schedule)

    rows = []
    for i, idx in enumerate(indices):
        img_path = os.path.join(args.img_dir, "BGYImg%d.png" % idx)
        if not os.path.isfile(img_path):
            # Some datasets index by position rather than by file name.
            img_path = all_imgs[idx] if idx < len(all_imgs) else None
        if img_path is None or not os.path.isfile(img_path):
            print("  [skip] BGYImg%d.png not found" % idx)
            continue

        bgr = load_image(img_path)
        img_t = img_to_tensor(bgr, device)

        if labels is not None and idx < len(labels):
            label_v = torch.tensor(labels[idx], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            label_v = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

        print("  [%d/%d] idx=%d  label=%s"
              % (i + 1, len(indices), idx, label_v.cpu().numpy().flatten().round(2)),
              end="  ", flush=True)

        gen_nocfg, gen_cfg = sample_one(
            model=model,
            diffusion=diffusion,
            ae=ae,
            struct_encoder=struct_encoder,
            img_tensor=img_t,
            label=label_v,
            scale_factor=scale_factor,
            struct_scale_factor=struct_scale_factor,
            struct_ch_indices=struct_ch_indices,
            cfg_scale=args.cfg_scale,
            latent_size=latent_size,
            device=device,
            dfa_cfg=dfa_cfg,
            is_flow=is_flow,
            ddim_eta=args.ddim_eta,
            seed=args.seed + i,
        )
        print("done")

        # BGR channels: 0 = nucleus (B), 1 = protein (G), 2 = microtubule (R).
        nucleus_u8 = colorize_gray(bgr[:, :, 0], (255, 80, 80))
        microtub_u8 = colorize_gray(bgr[:, :, 2], (80, 80, 255))
        protein_gt_u8 = colorize_gray(bgr[:, :, 1], (80, 255, 80))
        gen_nocfg_c = colorize_gray(gen_nocfg, (80, 255, 80))
        gen_cfg_c = colorize_gray(gen_cfg, (80, 255, 80))

        pad = np.full((IMG_SIZE, 2, 3), 40, dtype=np.uint8)
        panels = [nucleus_u8, pad, microtub_u8, pad, gen_nocfg_c, pad, gen_cfg_c, pad, protein_gt_u8]
        row_img = np.concatenate(panels, axis=1)

        titles = ["Nucleus(struct)", "", "Microtub(struct)", "", "Generated(no-cfg)", "",
                  cfg_title, "", "GT Protein"]
        header = make_label_row(panels, titles)
        row_with_header = np.concatenate([header, row_img], axis=0)

        label_str = "label=[%.2f, %.2f]  idx=%d" % (label_v[0, 0].item(), label_v[0, 1].item(), idx)
        cv2.putText(row_with_header, label_str, (4, row_with_header.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 50), 1, cv2.LINE_AA)

        rows.append(row_with_header)
        cv2.imwrite(os.path.join(args.out_dir, "compare_%05d.png" % idx), row_with_header)

    if not rows:
        print("no sample was generated")
        return

    combined = np.concatenate(rows, axis=0)
    out_path = os.path.join(
        args.out_dir, "cfg_compare_cfg%s_steps%d.png" % (args.cfg_scale, args.ddim_steps)
    )
    cv2.imwrite(out_path, combined)
    print("\n[done] grid saved to %s  shape=%s" % (out_path, combined.shape))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare no-CFG vs DFA-CFG sampling")
    p.add_argument("--dit-ckpt", type=str,
                   default="results/dit_x0_adv_cynp_v2/002-DiT-B-4/checkpoints/0010000.pt")
    p.add_argument("--main-ae-ckpt", type=str, default="")
    p.add_argument("--hpa-vae-ckpt", type=str, default="")
    p.add_argument("--img-dir", type=str,
                   default="")
    p.add_argument("--labels-h5", type=str,
                   default="")
    p.add_argument("--cfg-scale", type=float, default=1.25)

    # DFA-CFG (Dynamic Frequency-Aware CFG)
    p.add_argument("--no-dfa-cfg", action="store_true",
                   help="disable frequency weighting and fall back to plain CFG")
    p.add_argument("--dfa-w-low", type=float, default=None,
                   help="low-band CFG weight (default: --cfg-scale)")
    p.add_argument("--dfa-w-high", type=float, default=1.0,
                   help="high-band CFG weight (1.0 leaves high frequencies untouched)")
    p.add_argument("--dfa-w-low-min", type=float, default=1.0,
                   help="lower bound of the low-band weight schedule")
    p.add_argument("--dfa-w-high-min", type=float, default=1.0,
                   help="lower bound of the high-band weight schedule")
    p.add_argument("--dfa-mode", type=str, default="fft", choices=["fft", "bilinear"],
                   help="frequency decomposition: circular FFT low-pass or bilinear")
    p.add_argument("--dfa-ratio", type=float, default=0.3,
                   help="low-band radius relative to min(H, W) / 2")
    p.add_argument("--dfa-ratio-min", type=float, default=None,
                   help="low-band radius at the noise end for *ratio_anneal schedules")
    p.add_argument("--dfa-schedule", type=str, default="dual_linear",
                   choices=["legacy", "static", "dual_linear", "dual_cosine",
                            "ratio_anneal", "fresca", "fresca_ratio_anneal"],
                   help="temporal schedule of band width and weights")
    p.add_argument("--dfa-time-schedule", action=argparse.BooleanOptionalAction,
                   default=True, help="enable the temporal weight schedule")

    p.add_argument("--ddim-steps", type=int, default=250)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--n-samples", type=int, default=8,
                   help="number of randomly drawn images (ignored with --indices)")
    p.add_argument("--indices", type=str, default="",
                   help="comma-separated image indices, e.g. 0,100,500,1000")
    p.add_argument("--out-dir", type=str, default="./cfg_compare")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
