"""
Evaluate generation quality of a DiT checkpoint on the test split (CYPM/test by
default; use --data-root to point at CYNP_ALL and other datasets).

Protocol (paper Sec. 2.3), per test image:
  1. Condition on its struct channels (nucleus + microtubule, 2ch) and label.
  2. Draw n_per_sample samples (default 10) with different random noise.
  3. SSIM / PSNR: first generated sample vs. the real protein channel.
  4. LPIPS: all pairs among the n_per_sample images (diversity, higher is better).
  5. FID: all fakes vs. all reals (Inception-2048).
  6. Density / Coverage: VGG16 classifier[:4] 4096-d features (same as ProLDM), k-NN=10.

Implementation notes:
  - Architecture is rebuilt from the training config in ckpt['args']
    (seq_concat only; legacy channel_concat checkpoints raise).
  - scale_factor comes from the checkpoint; the AE is the training-time HPA VAE.
  - Inception input follows the ProLDM PNG convention: (R,G,B)=(0, protein in [0,1], 0).
  - All metrics are computed in image space (decoded [-1,1] -> [0,1]).

Usage:
  python scripts/eval_dit_metrics.py \\
      --ckpt ./outputs \\
      --num-samples 200 --n-per-sample 10 --ddim-steps 250 \\
      --save-all-images
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
import re
import sys
import time

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_metric
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from scipy import linalg  # noqa: F401  (used internally by calculate_frechet_distance)
from tqdm import tqdm

# ---- Make local DiT-main modules importable ----
_DIT_MAIN_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _DIT_MAIN_ROOT not in sys.path:
    sys.path.insert(0, _DIT_MAIN_ROOT)
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from dit_model_build import build_dit_seq_from_args  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from models import prepare_cfg_ddim_sampling, take_cfg_protein_latent  # noqa: E402
from fid_inception_weights import ensure_fid_inception_weights  # noqa: E402
from struct_encoder import load_struct_encoder  # noqa: E402
from utils.dynamic_freq_sampling import DFACFGConfig, DFACFGWrapper  # noqa: E402
from struct_hpa_encoder import (  # noqa: E402
    load_hpa_struct_encoder,
    load_hpa_main_ae,
    DEFAULT_HPA_CKPT,
)
from metrics_utils import (  # noqa: E402
    compute_density_coverage,
    LPIPSDiversityEvaluator,
    VGG16FeatureAccumulator,
)

IMG_SIZE = 256


# ============================================================
# Model & AE loading
# ============================================================

def build_dit_from_args(args_ns, device: torch.device):
    return build_dit_seq_from_args(args_ns, device)


def load_everything(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_ns = ckpt["args"]
    print(f"[ckpt] loaded {ckpt_path}")
    print(f"[ckpt] train config: model={args_ns.model}  inject_mode={getattr(args_ns, 'struct_inject_mode', 'seq_concat')}  "
          f"main_ae=hpa_vae  cfg_train_dprob={args_ns.cfg_train_dprob}  "
          f"minsnr_gamma={args_ns.minsnr_gamma}")

    # --- DiT ---
    model, main_L, struct_L = build_dit_from_args(args_ns, device)
    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
        print("[ckpt] using EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("[ckpt] using main model weights (no EMA)")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # --- struct_encoder: rebuild per the training-time backend ---
    struct_backend = getattr(args_ns, "struct_backend", "cnn")
    if struct_backend == "hpa_vae":
        hpa_ckpt = getattr(args_ns, "hpa_vae_ckpt", "") or DEFAULT_HPA_CKPT
        struct_encoder = load_hpa_struct_encoder(hpa_ckpt, device, latent_size=struct_L)
        print(f"[ckpt] struct backend=hpa_vae (L={struct_L}), rebuilt from {hpa_ckpt}")
    else:
        struct_ckpt_path = getattr(args_ns, "struct_encoder_ckpt", "") or None
        struct_encoder = load_struct_encoder(
            struct_ckpt_path, device, torch.float32, freeze=True,
        )
        if "struct_encoder" in ckpt:
            struct_encoder.load_state_dict(ckpt["struct_encoder"])
            print("[ckpt] struct backend=cnn, struct_encoder weights loaded")
        else:
            print("[ckpt] warning: cnn backend but ckpt has no struct_encoder key; random init")
    struct_encoder.eval()
    for p in struct_encoder.parameters():
        p.requires_grad = False

    hpa_main_ckpt = os.path.abspath(
        os.path.expanduser(
            (getattr(args_ns, "main_ae_ckpt", "") or "").strip()
            or getattr(args_ns, "main_ae_ckpt_resolved", "")
            or DEFAULT_HPA_CKPT
        )
    )
    ae = load_hpa_main_ae(hpa_main_ckpt, device)
    print(f"[main-ae] HPA VAE loaded {hpa_main_ckpt}")

    scale_factor = float(ckpt.get("scale_factor", 1.0))
    struct_scale_factor = float(ckpt.get("struct_scale_factor", 1.0))
    print(
        f"[scale] protein={scale_factor:.6f}  struct={struct_scale_factor:.6f}  "
        f"main_latent_size={main_L}"
    )

    return model, struct_encoder, ae, scale_factor, struct_scale_factor, main_L, args_ns


def decode_protein(ae, z_scaled: torch.Tensor, scale_factor: float) -> torch.Tensor:
    z = z_scaled / scale_factor
    decoded = ae.decode(z)
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    return decoded  # (B,1,256,256) in ~[-1,1]


# ============================================================
# Val data loading (file order; partial samples keep raw h5 values, as in training)
# ============================================================

def _read_image_3ch(fpath: str) -> torch.Tensor:
    """BGR PNG -> (3,H,W) float in [-1,1]."""
    img = cv2.imread(fpath)
    if img is None:
        raise FileNotFoundError(fpath)
    img = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1))


def _load_val_samples(
    val_dir: str,
    labels_h5: str,
    num_samples: int,
    struct_ch_indices,
    device: torch.device,
):
    """Returns [(idx, fpath, y), ...]; images are re-read per batch to keep memory flat."""
    import h5py

    with h5py.File(labels_h5, "r") as f:
        if "labels" in f:
            labels = f["labels"][:]
        elif "RealY" in f:
            labels = f["RealY"][:]
        else:
            raise KeyError(
                f"neither labels nor RealY in {labels_h5} (keys={list(f.keys())})"
            )

    files = sorted(glob.glob(os.path.join(val_dir, "BGYImg*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(val_dir, "BGImg*.bmp")))
    if num_samples and num_samples > 0:
        files = files[:num_samples]
    if not files:
        raise FileNotFoundError(
            f"no BGYImg*.png or BGImg*.bmp in {val_dir}"
        )

    out = []
    for fpath in files:
        base = os.path.basename(fpath)
        if base.startswith("BGImg") and base.endswith(".bmp"):
            idx = int(base.replace("BGImg", "").replace(".bmp", ""))
        else:
            idx = int(base.replace("BGYImg", "").replace(".png", ""))
        if idx < 0 or idx >= len(labels):
            raise IndexError(
                f"sample idx={idx} out of range (labels length {len(labels)}): {fpath}"
            )
        y_tensor = torch.from_numpy(labels[idx].astype(np.float32))
        out.append((idx, fpath, y_tensor))
    return out


def _batch_images_3ch(batch) -> torch.Tensor:
    return torch.stack([_read_image_3ch(s[1]) for s in batch], dim=0)


# ============================================================
# Metrics
# ============================================================

def _to_rgb01_for_fid(x_protein_batch: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) in [-1,1] -> (B,3,H,W) RGB float32 in [0,1], matching the ProLDM FID pipeline.

    ProLDM writes protein into the middle (green) PNG channel with R=B=0;
    The PIL RGB conversion plus ToTensor() gives the same layout.
    The pytorch_fid Inception still resizes/normalizes internally.
    """
    g = ((x_protein_batch + 1.0) / 2.0).clamp(0.0, 1.0).float()
    bs, _, h, w = g.shape
    out = torch.zeros(bs, 3, h, w, device=g.device, dtype=g.dtype)
    out[:, 1:2, :, :] = g
    return out


class _FIDAccumulator:
    """Streaming FID accumulator over pytorch_fid.InceptionV3.

    Each batch's features are moved to CPU to avoid holding everything on device.
    mu/sigma are estimated with numpy then passed to calculate_frechet_distance.
    """

    def __init__(
        self,
        feature_dim: int,
        device: torch.device,
        fid_mirror: str | None = None,
        fid_weights_path: str | None = None,
        fid_weights_url: str | None = None,
    ):
        ensure_fid_inception_weights(
            weights_path=fid_weights_path,
            url=fid_weights_url,
            mirror=fid_mirror,
        )
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim]
        self.net = InceptionV3([block_idx], resize_input=True, normalize_input=True).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.device = device
        self._real: list[np.ndarray] = []
        self._fake: list[np.ndarray] = []

    @torch.no_grad()
    def _extract(self, x_rgb_01: torch.Tensor) -> np.ndarray:
        # resize_input=True resizes to 299; normalize_input=True maps [0,1]->[-1,1].
        feats = self.net(x_rgb_01.to(self.device))[0]
        if feats.ndim == 4:
            feats = feats.squeeze(-1).squeeze(-1)
        return feats.cpu().numpy()

    def update(self, x_protein_batch: torch.Tensor, real: bool):
        feats = self._extract(_to_rgb01_for_fid(x_protein_batch))
        (self._real if real else self._fake).append(feats)

    @staticmethod
    def _stats(feats_list):
        feats = np.concatenate(feats_list, axis=0).astype(np.float64)
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma

    def compute(self) -> float:
        mu_r, sigma_r = self._stats(self._real)
        mu_f, sigma_f = self._stats(self._fake)
        return float(calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f))


def _protein_tensor_to_u8_png(t_1hw: torch.Tensor) -> np.ndarray:
    """(1,H,W) in [-1,1] on CPU → uint8(H,W) for cv2.imwrite."""
    return ((t_1hw[0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def compute_ssim_psnr(real: torch.Tensor, fake: torch.Tensor) -> tuple:
    """real, fake: (B,1,H,W) in [-1,1]. Returns (ssim_batch_np, psnr_batch_np)."""
    r01 = ((real + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    f01 = ((fake + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    ssim_vals, psnr_vals = [], []
    for b in range(r01.shape[0]):
        ssim_vals.append(ssim_metric(r01[b, 0], f01[b, 0], data_range=1.0))
        mse = float(((r01[b, 0] - f01[b, 0]) ** 2).mean())
        psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
        psnr_vals.append(psnr)
    return np.array(ssim_vals), np.array(psnr_vals)


# ============================================================
# Main
# ============================================================

def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    if getattr(args, "data_root", None) and str(args.data_root).strip():
        from train import resolve_labels_h5_path

        dr = os.path.normpath(str(args.data_root).strip())
        tag = (getattr(args, "dataset_h5_tag", None) or "CyPM").strip()
        args.val_dir = os.path.join(dr, "test")
        h5_path, h5_key = resolve_labels_h5_path(dr, "test", tag, required=True)
        args.val_labels_h5 = h5_path
        print(
            f"[data] from --data-root: {dr}  "
            f"(test + {os.path.basename(h5_path)}[{h5_key}])"
        )

    model, struct_encoder, ae, scale_factor, struct_scale_factor, main_L, train_args = load_everything(
        args.ckpt, device
    )

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

    _diff_type = getattr(train_args, "diffusion_type", "ddpm")
    _noise_schedule = getattr(train_args, "noise_schedule", "linear")
    if _diff_type == "flow":
        _flow_sigma = float(getattr(train_args, "flow_sigma_min", 0.0))
        _flow_heun = bool(getattr(train_args, "flow_use_heun", False))
        _flow_shift = float(getattr(train_args, "flow_time_shift_mu", 0.0))
        diffusion_ddim = create_diffusion(
            timestep_respacing=str(args.ddim_steps),
            learn_sigma=False,
            diffusion_type="flow",
            flow_num_steps=args.ddim_steps,
            flow_sigma_min=_flow_sigma,
            flow_use_heun=_flow_heun,
            flow_time_shift_mu=_flow_shift,
        )
        _diffusion_note = (
            f"flow steps={args.ddim_steps} heun={_flow_heun} shift={_flow_shift}"
        )
    else:
        diffusion_ddim = create_diffusion(
            timestep_respacing=str(args.ddim_steps),
            learn_sigma=False,
            predict_xstart=_predict_xstart,
            noise_schedule=_noise_schedule,
            diffusion_type="ddpm",
        )
        _diffusion_note = (
            f"ddpm/ddim steps={args.ddim_steps} predict_xstart={_predict_xstart} "
            f"noise_schedule={_noise_schedule}"
        )
    _cli_clamp = getattr(args, "ddim_x0_clamp", None)
    if _cli_clamp is None:
        _x0_clamp = float(getattr(train_args, "ddim_x0_clamp", 4.0) or 0.0)
        _clamp_src = "ckpt"
    else:
        _x0_clamp = float(_cli_clamp)
        _clamp_src = "cli"
    denoised_fn = (
        (lambda x, _c=_x0_clamp: x.clamp(-_c, _c)) if _x0_clamp > 0 else None
    )
    print(
        f"[diffusion] {_diffusion_note}, eta={args.ddim_eta}, cfg_scale={args.cfg_scale}, "
        f"ddim_x0_clamp={_x0_clamp} ({_clamp_src})"
    )
    if getattr(args, "cfg_freq_weighted", False):
        print(
            f"[facfg] sched={getattr(args, 'cfg_freq_schedule', 'legacy')} "
            f"w_low={getattr(args, 'cfg_w_low', args.cfg_scale)} "
            f"w_high={getattr(args, 'cfg_w_high', 1.0)} "
            f"wl_min={getattr(args, 'cfg_w_low_min', 1.0)} "
            f"wh_min={getattr(args, 'cfg_w_high_min', 1.0)} "
            f"mode={getattr(args, 'cfg_freq_mode', 'fft')} "
            f"ratio={getattr(args, 'cfg_freq_ratio', 0.3)} "
            f"tsched={getattr(args, 'cfg_freq_time_schedule', True)}"
        )
    if getattr(args, "cfg_zero_star", False):
        _zis = getattr(args, "cfg_zero_init_steps", None)
        _zss = getattr(args, "cfg_zero_star_scale", "band")
        print(
            f"[cfg-zero*] init_steps={_zis if _zis is not None else 'auto(4%N)'} "
            f"scale={_zss}"
            + (" + facfg" if getattr(args, "cfg_freq_weighted", False) else "")
        )

    use_cfg = args.cfg_scale > 1.0 + 1e-6
    if use_cfg and float(getattr(train_args, "cfg_train_dprob", 0.0)) <= 0.0:
        print(
            f"  [warn] cfg_scale={args.cfg_scale} but ckpt cfg_train_dprob=0; "
            "the unconditional branch may be undertrained"
        )
    if _predict_xstart:
        print("[dit] clean x0 mode: DDIM interprets the model output as START_X")

    # --- Load val data ---
    val_samples = _load_val_samples(
        args.val_dir,
        args.val_labels_h5,
        args.num_samples,
        train_args.struct_ch_indices,
        device,
    )
    N = len(val_samples)
    print(f"[data] {N} val images, will generate {args.n_per_sample} samples each "
          f"→ {N * args.n_per_sample} fake images total")

    fid = _FIDAccumulator(
        feature_dim=2048,
        device=device,
        fid_mirror=getattr(args, "fid_inception_mirror", None),
        fid_weights_path=getattr(args, "fid_inception_weights", None),
        fid_weights_url=getattr(args, "fid_inception_url", None),
    )
    vgg_feats = VGG16FeatureAccumulator(device=device)
    print("[vgg16] Density/Coverage extractor loaded (ProLDM classifier[:4], 4096-d)")

    lpips_eval = None
    if args.n_per_sample >= 2 and not getattr(args, "skip_lpips", False):
        lpips_eval = LPIPSDiversityEvaluator(device=device, net=args.lpips_net)
        print(f"[lpips] net={args.lpips_net}; pairwise LPIPS over {args.n_per_sample} samples per test image")
    elif args.n_per_sample < 2:
        print("[lpips] skipped: --n-per-sample < 2 cannot compute LPIPS diversity")

    # idx -> list of (1,H,W) CPU tensors in [-1,1] for LPIPS diversity
    fakes_by_idx: dict[int, list[torch.Tensor]] = {}

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis_first")
    os.makedirs(vis_dir, exist_ok=True)

    real_save_dir = fake_save_dir = ""
    if getattr(args, "save_all_images", False):
        real_save_dir = os.path.join(args.out_dir, "real")
        fake_save_dir = os.path.join(args.out_dir, "fake")
        os.makedirs(real_save_dir, exist_ok=True)
        os.makedirs(fake_save_dir, exist_ok=True)
        print(f"[save] real → {real_save_dir}  fake → {fake_save_dir}")

    # --- Push all real protein channels through FID as the real distribution ---
    print("[fid] pushing real images ...")
    for start in tqdm(range(0, N, args.gen_batch_size), desc="real"):
        batch = val_samples[start:start + args.gen_batch_size]
        img_3ch = _batch_images_3ch(batch).to(device)
        real_protein = img_3ch[:, 1:2]
        fid.update(real_protein, real=True)
        vgg_feats.update(real_protein, real=True)
        if real_save_dir:
            for i, (idx, _, _) in enumerate(batch):
                u8 = _protein_tensor_to_u8_png(img_3ch[i : i + 1, 1:2].cpu()[0])
                cv2.imwrite(os.path.join(real_save_dir, f"{idx:05d}.png"), u8)

    # --- Generation ---
    per_row: list[dict] = []
    ssim_first_all: list[float] = []
    psnr_first_all: list[float] = []

    # Outer loop over test-sample batches, inner loop over n_per_sample draws.
    gen_batch = args.gen_batch_size
    total_batches = (N + gen_batch - 1) // gen_batch
    t0 = time.time()

    for k_rep in range(args.n_per_sample):
        print(f"\n[gen] sampling round {k_rep + 1}/{args.n_per_sample}")
        pbar = tqdm(range(0, N, gen_batch), total=total_batches, desc=f"rep{k_rep + 1}")
        for start in pbar:
            batch = val_samples[start:start + gen_batch]
            b = len(batch)
            img_3ch = _batch_images_3ch(batch).to(device)  # (b,3,256,256)
            labels = torch.stack([s[2] for s in batch], dim=0).to(device)      # (b, 2)
            real_protein = img_3ch[:, 1:2]                                      # (b,1,256,256)

            c0, c1 = int(train_args.struct_ch_indices[0]), int(train_args.struct_ch_indices[1])
            s2 = img_3ch[:, [c0, c1]]
            z_struct = struct_encoder(s2) * struct_scale_factor

            # Fresh noise per round (seed offset by round index).
            gen_seed = args.seed + 1000 * k_rep + start
            g = torch.Generator(device=device).manual_seed(gen_seed)
            noise = torch.randn(b, 4, main_L, main_L, device=device, generator=g)

            sample_shape, noise_in, mk, loop_model, batch_use_cfg = prepare_cfg_ddim_sampling(
                b, noise, z_struct, labels, args.cfg_scale, _diff_type, model
            )
            if getattr(args, "cfg_freq_weighted", False):
                _dfa_cfg = DFACFGConfig(
                    enabled=True,
                    freq_mode=str(getattr(args, "cfg_freq_mode", "fft")),
                    freq_ratio=float(getattr(args, "cfg_freq_ratio", 0.3)),
                    freq_ratio_min=getattr(args, "cfg_freq_ratio_min", None),
                    freq_schedule=str(getattr(args, "cfg_freq_schedule", "legacy")),
                    time_schedule=bool(getattr(args, "cfg_freq_time_schedule", True)),
                    w_low=float(getattr(args, "cfg_w_low", args.cfg_scale)),
                    w_high=float(getattr(args, "cfg_w_high", 1.0)),
                    w_low_min=float(getattr(args, "cfg_w_low_min", 1.0)),
                    w_high_min=float(getattr(args, "cfg_w_high_min", 1.0)),
                )
                if str(_diff_type).lower() == "flow":
                    # FlowMatching performs the frequency-aware mixing inside the ODE loop.
                    mk["cfg_freq_weighted"] = True
                    mk["cfg_w_low"] = float(_dfa_cfg.low_weight(args.cfg_scale))
                    mk["cfg_w_high"] = _dfa_cfg.w_high
                    mk["cfg_freq_mode"] = _dfa_cfg.freq_mode
                    mk["cfg_freq_ratio"] = _dfa_cfg.freq_ratio
                    if _dfa_cfg.freq_ratio_min is not None:
                        mk["cfg_freq_ratio_min"] = _dfa_cfg.freq_ratio_min
                    mk["cfg_freq_time_schedule"] = _dfa_cfg.time_schedule
                    mk["cfg_freq_schedule"] = _dfa_cfg.freq_schedule
                    mk["cfg_w_low_min"] = _dfa_cfg.w_low_min
                    mk["cfg_w_high_min"] = _dfa_cfg.w_high_min
                else:
                    # DDPM/x0: mix on the denoiser output and keep cfg_freq_* away from the DiT.
                    print("[facfg] ddpm path: frequency mixing via DFACFGWrapper")
                    loop_model = DFACFGWrapper(model, args.cfg_scale, _dfa_cfg)
            if getattr(args, "cfg_zero_star", False):
                mk["cfg_zero_star"] = True
                _zis = getattr(args, "cfg_zero_init_steps", None)
                if _zis is not None:
                    mk["cfg_zero_init_steps"] = int(_zis)
                mk["cfg_zero_star_scale"] = str(
                    getattr(args, "cfg_zero_star_scale", "band")
                )
            samples = diffusion_ddim.ddim_sample_loop(
                loop_model,
                shape=sample_shape,
                noise=noise_in,
                clip_denoised=False,
                model_kwargs=mk,
                denoised_fn=denoised_fn,
                progress=False,
                device=device,
                eta=args.ddim_eta,
            )
            z_out = take_cfg_protein_latent(samples, b, batch_use_cfg, _diff_type)

            x_fake = decode_protein(ae, z_out[:, :4], scale_factor)  # (b,1,256,256) in [-1,1]

            if real_save_dir and fake_save_dir:
                for i, (idx, _, _) in enumerate(batch):
                    fake_u8 = _protein_tensor_to_u8_png(x_fake[i : i + 1].cpu()[0])
                    if args.n_per_sample <= 1:
                        fake_name = f"{idx:05d}.png"
                    else:
                        fake_name = f"{idx:05d}_r{k_rep:02d}.png"
                    cv2.imwrite(os.path.join(fake_save_dir, fake_name), fake_u8)

            # FID / VGG16: every rep is added as a fake.
            fid.update(x_fake, real=False)
            vgg_feats.update(x_fake, real=False)

            if lpips_eval is not None:
                for i, (idx, _, _) in enumerate(batch):
                    fakes_by_idx.setdefault(idx, []).append(
                        x_fake[i : i + 1].cpu()[0].clone()
                    )

            # SSIM / PSNR: computed on rep 0 only.
            if k_rep == 0:
                ssim_b, psnr_b = compute_ssim_psnr(real_protein, x_fake)
                for i, (idx, _, _) in enumerate(batch):
                    per_row.append({
                        "idx": idx,
                        "ssim_first": float(ssim_b[i]),
                        "psnr_first": float(psnr_b[i]),
                    })
                ssim_first_all.extend(ssim_b.tolist())
                psnr_first_all.extend(psnr_b.tolist())

                # Also dump the first few visualizations on round 0.
                if args.save_vis > 0 and start < args.save_vis:
                    for i, (idx, _, _) in enumerate(batch):
                        if start + i >= args.save_vis:
                            break
                        real_np = _protein_tensor_to_u8_png(real_protein[i : i + 1].cpu()[0])
                        fake_np = _protein_tensor_to_u8_png(x_fake[i : i + 1].cpu()[0])
                        side = np.concatenate([real_np, fake_np], axis=1)
                        cv2.imwrite(os.path.join(vis_dir, f"{idx:05d}_real_fake.png"), side)

    elapsed = time.time() - t0
    print(f"\n[gen] all sampling done in {elapsed/60:.1f} min")

    # --- Aggregate FID / Density / Coverage ---
    print("[fid] computing FID ...")
    fid_val = fid.compute()

    print(f"[metrics] computing Density / Coverage (k={args.k_nn}, VGG16-4096, ProLDM) ...")
    real_feats = vgg_feats.stacked_features(real=True)
    fake_feats = vgg_feats.stacked_features(real=False)
    density_val, coverage_val = compute_density_coverage(
        real_feats, fake_feats, nearest_k=int(args.k_nn),
    )

    lpips_mean = float("nan")
    lpips_std = float("nan")
    lpips_n_pairs = 0
    if lpips_eval is not None and fakes_by_idx:
        print("[lpips] computing generation diversity LPIPS ...")
        lpips_mean, lpips_std, lpips_n_pairs = lpips_eval.mean_pairwise_diversity(
            fakes_by_idx
        )

    ssim_np = np.array(ssim_first_all, dtype=np.float64)
    psnr_np = np.array(psnr_first_all, dtype=np.float64)

    summary = {
        "ckpt": os.path.basename(args.ckpt),
        "n_test_samples": N,
        "n_per_sample": args.n_per_sample,
        "ddim_steps": args.ddim_steps,
        "noise_schedule": _noise_schedule if _diff_type != "flow" else "flow",
        "ddim_x0_clamp": _x0_clamp,
        "cfg_scale": args.cfg_scale,
        "cfg_freq_weighted": bool(getattr(args, "cfg_freq_weighted", False)),
        "cfg_w_low": float(getattr(args, "cfg_w_low", args.cfg_scale)),
        "cfg_w_high": float(getattr(args, "cfg_w_high", 1.0)),
        "cfg_w_low_min": float(getattr(args, "cfg_w_low_min", 1.0)),
        "cfg_w_high_min": float(getattr(args, "cfg_w_high_min", 1.0)),
        "cfg_freq_schedule": str(getattr(args, "cfg_freq_schedule", "legacy")),
        "cfg_freq_ratio": float(getattr(args, "cfg_freq_ratio", 0.3)),
        "cfg_zero_star": bool(getattr(args, "cfg_zero_star", False)),
        "cfg_zero_init_steps": getattr(args, "cfg_zero_init_steps", None),
        "cfg_zero_star_scale": str(getattr(args, "cfg_zero_star_scale", "band")),
        "ssim_mean": float(ssim_np.mean()),
        "ssim_std": float(ssim_np.std()),
        "psnr_mean": float(psnr_np.mean()),
        "psnr_std": float(psnr_np.std()),
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "lpips_n_pairs": lpips_n_pairs,
        "density": density_val,
        "coverage": coverage_val,
        "k_nn": int(args.k_nn),
        "fid": fid_val,
        "elapsed_min": elapsed / 60.0,
    }
    print("\n========= FINAL METRICS =========")
    for k, v in summary.items():
        print(f"  {k}: {v if isinstance(v, str) else f'{v:.4f}' if isinstance(v, float) else v}")
    print("================================")

    # --- Save ---
    with open(os.path.join(args.out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row[0].keys()))
        w.writeheader()
        w.writerows(per_row)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f"\n[csv] per-sample → {os.path.join(args.out_dir, 'per_sample.csv')}")
    print(f"[csv] summary   → {os.path.join(args.out_dir, 'summary.csv')}")
    if args.save_vis > 0:
        print(f"[vis] first {args.save_vis} visualizations -> {vis_dir}")
    if getattr(args, "save_all_images", False):
        print(f"[save] real → {os.path.join(args.out_dir, 'real')}")
        print(f"[save] fake → {os.path.join(args.out_dir, 'fake')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="")
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="If set, use <root>/test and <root>/testlabels_{--dataset-h5-tag}.h5, overriding the --val-dir / --val-labels-h5 defaults below.",
    )
    parser.add_argument(
        "--dataset-h5-tag",
        type=str,
        default="CyPM",
        help="Used with --data-root; must match the CyPM/CyNp tag in the filename when --val-labels-h5 is set manually.",
    )
    parser.add_argument("--val-dir", type=str,
                        default="",
                        help="Test image directory (used when --data-root is unset)")
    parser.add_argument("--val-labels-h5", type=str,
                        default="")
    parser.add_argument("--out-dir", type=str,
                        default="")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of test samples (0 = all)")
    parser.add_argument("--n-per-sample", type=int, default=10,
                        help="Independent samples per test image (FID/Density/Coverage/LPIPS; SSIM/PSNR use the first only)")
    parser.add_argument(
        "--k-nn",
        type=int,
        default=10,
        help="k for Density / Coverage nearest neighbours (paper default 10)",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="vgg",
        choices=["vgg", "alex", "squeeze"],
        help="LPIPS backbone (vgg matches ProLDM/PLocGAN)",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS diversity (avoids the lpips dependency and saves time)",
    )
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument(
        "--ddim-x0-clamp",
        type=float,
        default=2.0,
        help="Clip DDIM x0 to [-c,c]; 2.0 was best in the subset sweep; <=0 disables.",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-freq-weighted", action="store_true", help="Enable FA-CFG frequency-weighted sampling")
    parser.add_argument("--cfg-w-low", type=float, default=None, help="FA-CFG low-frequency guidance weight; default=cfg-scale")
    parser.add_argument("--cfg-w-high", type=float, default=1.0, help="FA-CFG high-frequency guidance weight")
    parser.add_argument("--cfg-freq-mode", type=str, default="fft", choices=["fft", "bilinear"])
    parser.add_argument("--cfg-freq-ratio", type=float, default=0.3)
    parser.add_argument(
        "--cfg-freq-time-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decay the FA-CFG high-frequency weight with t",
    )
    parser.add_argument(
        "--cfg-freq-schedule",
        type=str,
        default="legacy",
        choices=[
            "legacy", "static", "dual_linear", "dual_cosine",
            "ratio_anneal", "fresca", "fresca_ratio_anneal",
        ],
        help="FA-CFG schedule: legacy / static / dual_linear / dual_cosine / ratio_anneal / fresca",
    )
    parser.add_argument("--cfg-w-low-min", type=float, default=1.0)
    parser.add_argument("--cfg-w-high-min", type=float, default=1.0)
    parser.add_argument(
        "--cfg-freq-ratio-min",
        type=float,
        default=None,
        help="Low-frequency cutoff at t=0 for ratio_anneal / fresca_ratio_anneal (t=1 uses cfg-freq-ratio)",
    )
    parser.add_argument(
        "--cfg-zero-star",
        action="store_true",
        help="CFG-Zero*: tuned uncond scaling plus optional zero-init; combined with --cfg-freq-weighted it becomes FA-CFG*",
    )
    parser.add_argument(
        "--cfg-zero-init-steps",
        type=int,
        default=None,
        help="Zero out v for the first ODE steps of CFG-Zero*; default max(1, round(0.04*N))",
    )
    parser.add_argument(
        "--cfg-zero-star-scale",
        type=str,
        default="band",
        choices=["band", "global"],
        help="st* in FA-CFG*: band = separate st* per band; global = one st* for the whole velocity field",
    )
    parser.add_argument(
        "--dit-parameterization",
        type=str,
        default="auto",
        choices=["auto", "eps", "x0"],
        help="auto = read from ckpt args; x0 or eps forces a value (must match the trained weights)",
    )
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--save-vis", type=int, default=16,
                        help="Save real/fake side-by-side PNGs for the first N (rep 0 only)")
    parser.add_argument(
        "--save-all-images",
        action="store_true",
        help="Write every real protein image to out-dir/real/ and every fake to out-dir/fake/. "
             "With n-per-sample>1 filenames are {idx:05d}_r{rep:02d}.png, otherwise {idx:05d}.png (matching real)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--fid-inception-mirror",
        type=str,
        default="auto",
        help="FID Inception weight download mirror: auto (try ghfast/ghproxy then GitHub), "
        "direct, ghfast, ghproxy, ghproxy_cc; env var FID_INCEPTION_MIRROR also works",
    )
    parser.add_argument(
        "--fid-inception-weights",
        type=str,
        default="",
        help="Existing local pt_inception-*.pth copied into the torch hub cache (no download)",
    )
    parser.add_argument(
        "--fid-inception-url",
        type=str,
        default="",
        help="Custom direct weights URL (takes precedence over mirror); or set FID_INCEPTION_URL",
    )
    args = parser.parse_args()
    if args.cfg_w_low is None:
        args.cfg_w_low = float(args.cfg_scale)
    main(args)
