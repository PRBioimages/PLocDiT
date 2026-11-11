#!/usr/bin/env python3
"""
Quick sampling test from the latest DiT checkpoint using val set images.
"""

import os as _os
import sys as _sys
_SUBDIT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_SUBDIT_ROOT, _os.path.join(_SUBDIT_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import cv2
import argparse
import os
import glob
from torchvision.utils import save_image

from diffusion import create_diffusion
from struct_encoder import load_struct_encoder
from dit_model_build import build_dit_seq_from_args

from struct_hpa_encoder import load_hpa_main_ae, load_hpa_struct_encoder, DEFAULT_HPA_CKPT
IMG_SIZE = 256


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ae_ckpt = os.path.abspath(os.path.expanduser((args.main_ae_ckpt or "").strip() or DEFAULT_HPA_CKPT))
    if not os.path.isfile(ae_ckpt):
        raise FileNotFoundError(f"HPA VAE weights not found: {ae_ckpt}")
    ae = load_hpa_main_ae(ae_ckpt, torch.device(device))
    print(f"Loaded HPA VAE from {ae_ckpt}")

    ckpt_full = torch.load(args.dit_ckpt, map_location=device, weights_only=False)
    train_args = ckpt_full.get("args")
    if train_args is None:
        raise ValueError("Checkpoint has no 'args'; cannot rebuild the seq_concat DiT.")
    model, latent_size, _ = build_dit_seq_from_args(train_args, torch.device(device))

    struct_backend = str(getattr(train_args, "struct_backend", "cnn") or "cnn")
    if struct_backend == "hpa_vae":
        _hpa_struct_L = int(getattr(train_args, "hpa_struct_latent_size", 0) or 0)
        if _hpa_struct_L <= 0:
            _hpa_struct_L = latent_size
        _hpa_ckpt = os.path.abspath(os.path.expanduser(
            (getattr(args, "hpa_vae_ckpt", "") or "").strip() or DEFAULT_HPA_CKPT))
        struct_encoder = load_hpa_struct_encoder(
            ckpt_path=_hpa_ckpt,
            device=torch.device(device),
            latent_size=_hpa_struct_L,
        )
        print(f"Struct encoder: backend=hpa_vae L={_hpa_struct_L} (frozen)")
    else:
        struct_encoder = load_struct_encoder(
            args.struct_encoder_ckpt if str(args.struct_encoder_ckpt).strip() else None,
            torch.device(device),
            torch.float32,
            freeze=True,
        )
        if ckpt_full.get("struct_encoder") is not None:
            struct_encoder.load_state_dict(ckpt_full["struct_encoder"], strict=True)
            print("Struct encoder: backend=cnn weights restored from the DiT checkpoint")
    cli_ss = float(getattr(args, "struct_scale_factor", 0.0) or 0.0)
    if cli_ss > 0:
        struct_scale_factor = cli_ss
    else:
        struct_scale_factor = float(ckpt_full.get("struct_scale_factor", 1.0))
    print(f"struct_scale_factor={struct_scale_factor:.6f}")
    scale_factor = float(ckpt_full.get("scale_factor", 1.0))
    print(f"scale_factor={scale_factor:.6f} (from checkpoint, default 1.0 if missing)")
    model.load_state_dict(ckpt_full["ema"])
    print(f"Loaded EMA weights from {args.dit_ckpt}")
    model.eval()
    print(f"DiT parameters: {sum(p.numel() for p in model.parameters()):,}")
    _cli_dp = getattr(args, "dit_parameterization", "auto").lower()
    _ckpt_dp = getattr(train_args, "dit_parameterization", "eps")
    _ckpt_is_x0 = _ckpt_dp == "x0"
    if _cli_dp == "auto":
        _predict_xstart = _ckpt_is_x0
    elif _cli_dp == "x0":
        _predict_xstart = True
    elif _cli_dp == "eps":
        _predict_xstart = False
    else:
        raise ValueError(f"unknown --dit-parameterization: {_cli_dp!r}")
    if _cli_dp != "auto" and (_predict_xstart != _ckpt_is_x0):
        print(
            f"[warn] CLI --dit-parameterization={_cli_dp} differs from ckpt dit_parameterization={_ckpt_dp!r}"
        )

    _aux = bool(getattr(train_args, "dit_predict_aux_x0", False)) and not _predict_xstart
    # DiT already exposes both heads when predict_aux_x0 is on.
    sample_denoiser = model
    if _predict_xstart:
        print("[dit] clean x0 sampling (START_X)")

    diffusion = create_diffusion(
        str(args.num_sampling_steps),
        learn_sigma=False,
        predict_xstart=_predict_xstart,
    )
    print(
        f"Diffusion: {args.num_sampling_steps} steps, sampler={args.sampler}, predict_xstart={_predict_xstart}"
    )

    # Load val images
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "BGYImg*.png")))[:args.num_images]
    print(f"Testing with {len(img_paths)} images")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, img_path in enumerate(img_paths):
        idx = int(os.path.basename(img_path).replace("BGYImg", "").replace(".png", ""))

        # RGB file semantics: R=microtubule, G=protein, B=nucleus; cv2 CHW order: 0=nucleus, 1=protein, 2=microtubule
        raw = cv2.imread(img_path)
        raw = raw.astype(np.float32) / 127.5 - 1.0
        raw = raw.transpose(2, 0, 1)
        struct_img = torch.from_numpy(raw).float().unsqueeze(0).to(device)

        ch0, ch1 = int(args.struct_ch_indices[0]), int(args.struct_ch_indices[1])
        struct_2ch = struct_img[:, [ch0, ch1], :, :].contiguous()
        z_struct = struct_encoder(struct_2ch) * struct_scale_factor

        # real protein channel for comparison
        real_protein = struct_img[:, 1:2]

        label = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

        # Noise
        z_noise = torch.randn(1, 4, latent_size, latent_size, device=device)

        model_kwargs = dict(z_struct=z_struct, y=label)

        t0 = __import__('time').time()
        with torch.no_grad():
            if args.sampler == "ddim":
                samples = diffusion.ddim_sample_loop(
                    sample_denoiser,
                    shape=(1, 4, latent_size, latent_size),
                    noise=z_noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                    eta=args.ddim_eta,
                )
            else:
                samples = diffusion.p_sample_loop(
                    sample_denoiser,
                    shape=(1, 4, latent_size, latent_size),
                    noise=z_noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                )
        elapsed = __import__('time').time() - t0

        z_protein = samples[:, :4]
        # Latents live in the scaled space; divide by scale_factor, then clamp before decode.
        z_protein = (z_protein / scale_factor).clamp(-6.0, 6.0)
        decoded = ae.decode(z_protein)
        x_protein = decoded.sample() if hasattr(decoded, 'sample') else decoded

        # Save generated
        save_image(
            x_protein,
            os.path.join(args.output_dir, f"gen_{idx:04d}.png"),
            normalize=True,
            value_range=(-1, 1),
        )
        # Save real protein for comparison
        save_image(
            real_protein,
            os.path.join(args.output_dir, f"real_{idx:04d}.png"),
            normalize=True,
            value_range=(-1, 1),
        )
        # Save side-by-side: real | gen
        compare = torch.cat([real_protein, x_protein], dim=3)  # (1,1,256,512)
        save_image(
            compare,
            os.path.join(args.output_dir, f"cmp_{idx:04d}.png"),
            normalize=True,
            value_range=(-1, 1),
        )

        print(f"  [{i+1}/{len(img_paths)}] idx={idx} ({elapsed:.1f}s)")

    print(f"\nDone! {len(img_paths)} images generated in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-ckpt", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cypm_results", "001-DiT-S-2", "checkpoints", "0021600.pt"))
    parser.add_argument(
        "--main-ae-ckpt",
        type=str,
        default="",
        help=f"HPA VAE; defaults to {DEFAULT_HPA_CKPT}",
    )
    parser.add_argument("--img-dir", type=str,
        default=os.path.join(os.path.expanduser("~"), "ProLDM_imgdata", "CYPM", "val"))
    parser.add_argument("--output-dir", type=str, default="./test_samples")
    parser.add_argument(
        "--struct-encoder-ckpt",
        type=str,
        default="",
        help="Structure encoder weights matching train.py",
    )
    parser.add_argument(
        "--struct-ch-indices",
        type=int,
        nargs=2,
        default=[0, 2],
        metavar=("CH0", "CH1"),
    )
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--struct-scale-factor",
        type=float,
        default=0.0,
        help="0 reads struct_scale_factor from the checkpoint (1.0 if absent)",
    )
    parser.add_argument(
        "--dit-parameterization",
        type=str,
        default="auto",
        choices=["auto", "eps", "x0"],
        help="auto reads the ckpt args; x0 forces clean x0 sampling (must match training)",
    )
    args = parser.parse_args()
    main(args)
