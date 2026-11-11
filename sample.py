# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sampling script for DiT + HPA VAE (CELL-Diff ``hpa_pretrained.bin`` latent space).

Usage:
    python sample.py \\
        --dit-ckpt results/.../checkpoints/0100000.pt \\
        --main-ae-ckpt pretrained/hpa_pretrained.bin \\
        --dit-parameterization auto \\
        --struct-img-dir /path/to/test/images \\
        --output-dir ./samples

    An empty ``--main-ae-ckpt`` falls back to struct_hpa_encoder.DEFAULT_HPA_CKPT.

    For checkpoints trained on clean x0 only (``train.py --dit-parameterization x0``), ``auto`` reads ckpt ``args``
    and samples from START_X; ``--dit-parameterization x0`` also works.
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
import pandas as pd
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
        raise ValueError("Checkpoint has no 'args' field; cannot rebuild the seq_concat DiT. Use a checkpoint saved by the current train.py.")
    model, latent_size, _struct_L = build_dit_seq_from_args(train_args, torch.device(device))

    # ======== Struct encoder: mirror the backend used at training time ========
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
        else:
            print("Struct encoder: backend=cnn (no weights in the checkpoint; random init)")
    cli_ss = float(getattr(args, "struct_scale_factor", 0.0) or 0.0)
    if cli_ss > 0:
        struct_scale_factor = cli_ss
        print(f"struct_scale_factor={struct_scale_factor:.6f} (CLI)")
    else:
        struct_scale_factor = float(ckpt_full.get("struct_scale_factor", 1.0))
        print(f"struct_scale_factor={struct_scale_factor:.6f} (from checkpoint, default 1.0 if missing)")
    scale_factor = float(ckpt_full.get("scale_factor", 1.0))
    print(f"scale_factor={scale_factor:.6f} (from checkpoint, default 1.0 if missing)")
    # Pick the weight dict; three save formats are supported
    if "ema" in ckpt_full and ckpt_full["ema"] is not None:
        sd = ckpt_full["ema"]
        tag = "EMA"
    elif "model" in ckpt_full:
        sd = ckpt_full["model"]
        tag = "model"
    else:
        sd = ckpt_full
        tag = "checkpoint"

    # Old checkpoints may lack null_struct_embed; fill missing keys from the fresh model
    _missing = set(model.state_dict().keys()) - set(sd.keys())
    if _missing:
        for k in sorted(_missing):
            if k.endswith("null_struct_embed"):
                print(f"[sample] old checkpoint lacks {k}, using the fresh init")
                sd[k] = model.state_dict()[k].clone()
            else:
                print(f"[sample] warning: missing key {k}, using the current model's init")
                sd[k] = model.state_dict()[k].clone()
    model.load_state_dict(sd)
    print(f"Loaded {tag} weights")
    model.eval()
    print(f"Loaded DiT from {args.dit_ckpt}")
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
            f"[warn] CLI --dit-parameterization={_cli_dp} differs from ckpt args.dit_parameterization={_ckpt_dp!r};"
            "check that it matches how the model was trained."
        )

    _aux = bool(getattr(train_args, "dit_predict_aux_x0", False)) and not _predict_xstart
    _diff_type = getattr(train_args, "diffusion_type", "ddpm")
    _is_flow = _diff_type == "flow"
    if _is_flow:
        # Flow matching: model directly predicts velocity, no TwinHeadEpsAdapter needed
        sample_denoiser = model
        print(f"[dit] Flow matching mode: velocity prediction, {getattr(train_args, 'flow_num_steps', 28)} steps")
    elif _aux:
        # DiT already returns both the eps and the aux-x0 heads when
        # predict_aux_x0 is on, so no adapter wrapper is needed.
        sample_denoiser = model
        print("[dit] dit_predict_aux_x0: sampling uses the eps head only")
    else:
        sample_denoiser = model
    if _predict_xstart and not _is_flow:
        print("[dit] clean x0 sampling (GaussianDiffusion START_X)")

    if _is_flow:
        diffusion = create_diffusion(
            str(args.num_sampling_steps),
            learn_sigma=False,
            predict_xstart=False,
            diffusion_type="flow",
            flow_num_steps=args.num_sampling_steps,
            flow_sigma_min=float(getattr(train_args, "flow_sigma_min",
                               getattr(args, "flow_sigma_min", 0.0))),
            flow_use_heun=bool(getattr(train_args, "flow_use_heun",
                               getattr(args, "flow_use_heun", False))),
            flow_time_shift_mu=float(getattr(train_args, "flow_time_shift_mu",
                               getattr(args, "flow_time_shift_mu", 0.0))),
        )
    else:
        diffusion = create_diffusion(
            str(args.num_sampling_steps),
            learn_sigma=False,
            predict_xstart=_predict_xstart,
        )
    print(
        f"Diffusion: {args.num_sampling_steps} steps, sampler={args.sampler}, "
        f"type={_diff_type}, predict_xstart={_predict_xstart}"
    )

    # ======== Load test images ========
    img_paths = sorted(glob.glob(os.path.join(args.struct_img_dir, "*.png")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(args.struct_img_dir, "*.jpg")))
    assert len(img_paths) > 0, f"No images found in {args.struct_img_dir}"
    print(f"Found {len(img_paths)} images")

    # Load labels CSV
    labels_dict = {}
    if args.label_csv and os.path.exists(args.label_csv):
        df = pd.read_csv(args.label_csv)
        for _, row in df.iterrows():
            idx = int(row['idx'])
            labels_dict[idx] = [float(row['p_cytoplasm']), float(row['p_nucleus'])]
        print(f"Loaded {len(labels_dict)} labels")

    os.makedirs(args.output_dir, exist_ok=True)

    # ======== Sample per image (batch=1) ========
    for i, img_path in enumerate(img_paths):
        idx = int(os.path.basename(img_path).replace("BGYImg", "").replace(".png", ""))

        # Read structure image: (3, 256, 256), range [-1, 1]
        # Standard RGB PNG: R=microtubule, G=protein, B=nucleus; cv2 gives CHW: ch0=nucleus, ch1=protein, ch2=microtubule
        raw = cv2.imread(img_path)
        raw = raw.astype(np.float32) / 127.5 - 1.0
        raw = raw.transpose(2, 0, 1)
        struct_img = torch.from_numpy(raw).float().unsqueeze(0).to(device)

        ch0, ch1 = int(args.struct_ch_indices[0]), int(args.struct_ch_indices[1])
        struct_2ch = struct_img[:, [ch0, ch1], :, :].contiguous()
        z_struct = struct_encoder(struct_2ch) * struct_scale_factor

        if labels_dict and idx in labels_dict:
            label = torch.tensor([labels_dict[idx]], dtype=torch.float32, device=device)
        else:
            label = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

        # Noise
        z_noise = torch.randn(1, 4, latent_size, latent_size, device=device)

        model_kwargs = dict(z_struct=z_struct, y=label)
        if _is_flow and getattr(args, "cfg_scale", 1.0) > 1.0:
            model_kwargs["cfg_scale"] = float(args.cfg_scale)

        with torch.no_grad():
            if _is_flow:
                # Flow matching: ODE solver (Euler/Heun)
                samples = diffusion.ddim_sample_loop(
                    sample_denoiser,
                    shape=(1, 4, latent_size, latent_size),
                    noise=z_noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                )
            elif args.sampler == "ddim":
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

        z_protein = samples[:, :4]

        # Decode
        with torch.no_grad():
            decoded = ae.decode(z_protein / scale_factor)
            x_protein = decoded.sample() if hasattr(decoded, "sample") and callable(decoded.sample) else decoded

        # Save
        save_image(
            x_protein,
            os.path.join(args.output_dir, f"gen_{idx:04d}.png"),
            normalize=True,
            value_range=(-1, 1),
        )

        if (i + 1) % 50 == 0:
            print(f"  Sampled {i + 1}/{len(img_paths)}")

    # Save grid
    out_files = sorted(glob.glob(os.path.join(args.output_dir, "gen_*.png")))
    if out_files:
        grid = torch.cat([torch.from_numpy(cv2.imread(f).transpose(2, 0, 1)).float() / 255.0
                          for f in out_files[:20]], dim=0)
        save_image(grid, os.path.join(args.output_dir, "grid_protein.png"),
                   nrow=args.nrow, normalize=True, value_range=(0, 1))
        print(f"Saved grid")

    print(f"\nDone! {len(img_paths)} images generated in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-ckpt", type=str, required=True)
    parser.add_argument(
        "--main-ae-ckpt",
        type=str,
        default="",
        help=f"HPA VAE weights; empty falls back to {DEFAULT_HPA_CKPT}",
    )
    parser.add_argument(
        "--hpa-vae-ckpt",
        type=str,
        default="",
        help="HPA VAE weights for the hpa_vae struct backend (empty -> pretrained/hpa_pretrained.bin)",
    )
    parser.add_argument("--struct-img-dir", type=str, required=True)
    parser.add_argument("--label-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./samples")
    parser.add_argument("--struct-encoder-ckpt", type=str, default="")
    parser.add_argument(
        "--struct-ch-indices",
        type=int,
        nargs=2,
        default=[0, 2],
        metavar=("CH0", "CH1"),
    )
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddpm")
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--struct-scale-factor",
        type=float,
        default=0.0,
        help="Struct latent multiplier; 0 reads struct_scale_factor from the checkpoint (1.0 for old checkpoints without it)",
    )
    parser.add_argument(
        "--dit-parameterization",
        type=str,
        default="auto",
        choices=["auto", "eps", "x0"],
        help=(
            "Same as train.py: whether GaussianDiffusion expects eps or a clean x0. "
            "auto reads ckpt['args'].dit_parameterization (recommended); "
            "x0 forces START_X (must match how the weights were trained)."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="CFG scale (>1 enables classifier-free guidance); only used in flow mode",
    )
    parser.add_argument(
        "--flow-sigma-min",
        type=float,
        default=0.0,
        help="Minimum sigma for flow-matching sampling (0 = integrate to zero noise); overrides the ckpt value",
    )
    parser.add_argument(
        "--flow-use-heun",
        action="store_true",
        help="Use the 2nd-order Heun solver for flow-matching sampling (Euler by default)",
    )
    parser.add_argument(
        "--flow-time-shift-mu",
        type=float,
        default=0.0,
        help="Flow-matching time-shift mu (Flux style); overrides the ckpt value",
    )
    args = parser.parse_args()
    main(args)
