# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Conditional DiT training in latent space (CYPM layout by default; CYNP / MTLS supported):
  - Data: --data-root points to a root holding train/ and val/; three-channel BGYImg*.png or two-channel BGImg*.bmp (MTLS).
    Labels h5: trainlabels_{tag}.h5 (key ``labels``) or LyMo2010Trainlabels_.h5 (key ``RealY``).
  - Main branch AE: frozen CELL-Diff HPA VAE (``hpa_pretrained.bin``) -> z of shape (B,4,L,L), L=64 (4x downsample).
  - Structure: CNN or HPA VAE (--struct-backend); the DiT side is always seq_concat (struct tokens appended to the sequence).
  - DiT: predicts the 4-channel protein eps by default (``--dit-parameterization eps``); ``--dit-parameterization x0`` predicts only the clean latent x0 (START_X) and conflicts with ``--dit-predict-aux-x0``.
    x0 mode optionally adds a **multi-scale latent residual** (``--dit-x0-ms-latent-coef``, avg_pool2d only, negligible extra VRAM).
  - Labels: 2-D float ``(p_cytoplasm, p_nucleus)`` from h5 ``labels``, injected into adaLN via a linear layer concatenated with the timestep embedding.
  - CFG: during training (y, z_struct) are dropped jointly with some probability; validation DDIM may use cfg_scale>1.
  - Validation loss uses val; DDIM grids default to fixed indices of the **test** split (``--vis-split test`` + ``--vis-val-indices``, falling back to val when test is missing); ``--vis-split val`` selects val explicitly; fixed train indices or a random train batch remain available via ``--vis-train-indices`` or an empty value; ``--vis-noise-seed`` defaults to 42 for fixed DDIM noise (``-1`` resamples each epoch). With ``--vis-seq-extra`` (on by default) each DDIM visualization appends one more hold-out sample after the fixed indices (epoch1->+0, epoch2->+1...).
  - Scale: protein latents are multiplied by scale_factor and struct latents by struct_scale_factor (estimated from the training-set std and cached by default).
  - Optional recon L1: ``--recon-l1-coef`` + ``--recon-l1-space image|latent`` (latent space skips AE.decode and saves VRAM).
  - Optional frequency loss: ``--recon-freq-coef`` applies high/low band MSE on the FFT magnitude spectra of x0_hat and the target (see ``freq_loss.py``).
  - Optional latent adversarial loss: ``--latent-adv-coef``>0. Default ``--latent-adv-mode lddgan``: on DDPM, discriminate the
    intermediate transition ``D(x_{t-1}, x_t, t, z_struct)`` (true posterior vs. fake posterior built from x0_hat) plus a quantitative label head (partial-label candidate selection);
    ``plocgan`` falls back to the endpoint ``D([z_struct, x0_hat])``. LDDGAN requires ``--diffusion-type ddpm``.
  - EMA: off by default; ``--ema`` keeps a shadow EMA of the trainable DiT parameters (decay=0.999 for the first ``ema_decay_switch_step`` steps, then 0.9995); checkpoints store ``ema`` and validation uses the EMA weights.
  - Resume: ``--resume path/to/NNNNNNN.pt`` restores model / optimizer / (if present) EMA and struct_encoder; the step count comes from the filename or the ``train_steps`` field; logs and new checkpoints still go to the original experiment directory.
"""

import os as _os
import sys as _sys
_PLOCDIT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_PLOCDIT_ROOT, _os.path.join(_PLOCDIT_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import os

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader, default_collate
import numpy as np
from glob import glob
import time
import argparse
import logging
import cv2

from models import DiT_models, DiTWithStructSeq
from struct_hpa_encoder import (
    load_hpa_struct_encoder,
    hpa_struct_out_channels,
    load_hpa_main_ae,
    hpa_main_latent_size,
    DEFAULT_HPA_CKPT,
)
from diffusion import create_diffusion
from diffusion.gaussian_diffusion import mean_flat, ModelMeanType
from freq_loss import mean_flat_freq_loss, mean_flat_laplacian_loss
from struct_encoder import load_struct_encoder
from latent_discriminator import (
    LatentLDDGANDiscriminator,
    LatentPlocGANDiscriminator,
    hinge_discriminator_loss,
    hinge_generator_loss,
    plocgan_discriminator_loss,
    plocgan_generator_loss,
    q_sample_posterior,
)

#################################################################################
#                             Training Helper Functions                           #
#################################################################################

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    pass


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt", mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def _parse_train_steps_from_ckpt_path(path: str) -> int:
    stem, _ = os.path.splitext(os.path.basename(path))
    try:
        return int(stem)
    except ValueError:
        return 0


def _resolve_hpa_weight_arg(
    cli_value: str | None,
    parser: argparse.ArgumentParser,
    arg_flag: str,
) -> str:
    """Resolve the HPA weight path: empty -> DEFAULT_HPA_CKPT; a missing explicit path falls back to the default if present (common with old hy-tmp paths)."""
    default_p = os.path.abspath(os.path.expanduser(DEFAULT_HPA_CKPT))
    raw = (cli_value or "").strip()
    if raw:
        primary = os.path.abspath(os.path.expanduser(raw))
        if os.path.isfile(primary):
            return primary
        if os.path.isfile(default_p):
            print(
                f"[train] warning: path given by {arg_flag} does not exist, using: {default_p}\n"
                f"         (invalid path: {raw})",
                flush=True,
            )
            return default_p
        parser.error(
            f"{arg_flag}: HPA VAE weights not found: {primary}\n"
            f" and the default weights are missing too: {default_p}\n"
            " Place hpa_pretrained.bin or set the environment variable HPA_VAE_PRETRAINED_CKPT."
        )
    if os.path.isfile(default_p):
        return default_p
    parser.error(
        f"Default HPA VAE weights not found: {default_p}\n"
        f"({arg_flag} not given; use --main-ae-ckpt / --hpa-vae-ckpt "
        "or the environment variable HPA_VAE_PRETRAINED_CKPT.)"
    )


#################################################################################
#                        Dataset root (CYPM / CYNP_ALL / …)                      #
#################################################################################

def _load_cypm_root():
    default = os.environ.get("PLOCDIT_DATA_ROOT", "")
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "dataset_cypm.yaml")
    if os.path.exists(yaml_path):
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(yaml_path)
            r = cfg.get("root", default)
            return str(r).strip() if r is not None else default
        except Exception:
            pass
    return default


# Fallback data root when --data-root is empty (config/ first, then this default)
_CYPM_ROOT = _load_cypm_root()
IMG_SIZE = 256
DEFAULT_VIS_NOISE_SEED = 42

# Candidate labels for partial (dual-localization) samples; matches the three middle bins of the test set
PARTIAL_LABEL_CANDIDATES = np.array(
    [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]], dtype=np.float32
)

_LYMO_LABEL_SPLIT = {"train": "Train", "val": "Val", "test": "Test"}


def resolve_labels_h5_path(
    data_root: str, split: str, h5tag: str, *, required: bool = True
) -> tuple[str, str]:
    """Resolve the labels h5 path and the dataset key. Returns (path, h5_key)."""
    tag = (h5tag or "").strip()
    tried: list[str] = []
    if tag:
        p = os.path.join(data_root, f"{split}labels_{tag}.h5")
        tried.append(p)
        if os.path.isfile(p):
            return p, "labels"
    lymo = os.path.join(
        data_root, f"LyMo2010{_LYMO_LABEL_SPLIT[split]}labels_.h5"
    )
    tried.append(lymo)
    if os.path.isfile(lymo):
        return lymo, "RealY"
    if not required:
        return "", ""
    raise FileNotFoundError(
        f"No {split} label h5 found under {data_root!r} (tried: {tried})."
        f" Check --data-root and --dataset-h5-tag (tag={tag!r})."
    )


def detect_image_format(data_root: str) -> str:
    """Infer the image format from filenames under train/: bgy3ch or bg2ch."""
    train_dir = os.path.join(data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Data directory does not exist: {train_dir}")
    if glob(os.path.join(train_dir, "BGImg*.bmp")):
        return "bg2ch"
    if glob(os.path.join(train_dir, "BGYImg*.png")):
        return "bgy3ch"
    raise FileNotFoundError(
        f"No BGYImg*.png (three-channel) or BGImg*.bmp (nucleus+protein) found under {train_dir}."
    )


def read_fluorescence_sample_chw(
    imgs_basedir: str, sample_idx: int, image_format: str
) -> np.ndarray:
    """Read one fluorescence sample as a CHW float32 array in [-1, 1] with shape (3, H, W).

    bgy3ch: BGYImg{idx}.png, ch0=nucleus, ch1=protein, ch2=third structure marker.
    bg2ch:  BGImg{idx}.bmp, ch0=nucleus, ch1=protein, ch2 is usually 0.
    """
    if image_format == "bg2ch":
        fname = f"BGImg{sample_idx}.bmp"
    elif image_format == "bgy3ch":
        fname = f"BGYImg{sample_idx}.png"
    else:
        raise ValueError(f"Unknown image_format: {image_format!r}")
    path = os.path.join(imgs_basedir, fname)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = img.astype(np.float32) / 127.5 - 1.0
    return img.transpose(2, 0, 1).copy()


def protein_channel(img: torch.Tensor, ch_index: int = 1) -> torch.Tensor:
    """Extract the single-channel protein image (B, 1, H, W) from (B, C, H, W)."""
    c = int(ch_index)
    return img[:, c : c + 1, :, :].contiguous()


def _load_pseudo_label_store(path: str, split: str, n_samples: int) -> dict:
    """Load a pseudo-label .h5 or .csv and return p_cyt/p_nuc/is_partial/use_pseudo arrays."""
    import csv
    import h5py

    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            if split not in f:
                raise KeyError(
                    f"pseudo-label h5 has no split={split!r} (found {list(f.keys())}): {path}"
                )
            g = f[split]
            p_cyt = g["p_cyt_final"][()].astype(np.float32)
            p_nuc = g["p_nuc_final"][()].astype(np.float32)
            is_partial = g["is_partial"][()].astype(bool)
            use_pseudo = g["use_pseudo"][()].astype(bool)
        if len(p_cyt) != n_samples:
            raise ValueError(
                f"pseudo-label h5 has {len(p_cyt)} samples but labels h5 has {n_samples}: {path}"
            )
        return {
            "p_cyt": p_cyt,
            "p_nuc": p_nuc,
            "is_partial": is_partial,
            "use_pseudo": use_pseudo,
        }

    if path.endswith(".csv"):
        p_cyt = np.full(n_samples, np.nan, dtype=np.float32)
        p_nuc = np.full(n_samples, np.nan, dtype=np.float32)
        is_partial = np.zeros(n_samples, dtype=bool)
        use_pseudo = np.zeros(n_samples, dtype=bool)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                idx = int(row["idx"])
                if idx < 0 or idx >= n_samples:
                    continue
                # Prefer continuous estimates (exclusive-soft reassignment), else fall back to legacy fields
                p_cyt[idx] = float(
                    row.get("p_cyt_ratio_est")
                    or row.get("p_cyt_pseudo")
                    or row.get("p_cyto")
                )
                p_nuc[idx] = float(
                    row.get("p_nuc_ratio_est")
                    or row.get("p_nuc_pseudo")
                    or row.get("p_nucleus")
                )
                is_partial[idx] = int(row.get("is_partial", 0)) == 1
                use_pseudo[idx] = int(row.get("use_pseudo", 1)) == 1
        return {
            "p_cyt": p_cyt,
            "p_nuc": p_nuc,
            "is_partial": is_partial,
            "use_pseudo": use_pseudo,
        }

    raise ValueError(f"Unsupported pseudo-label format (need .h5 or .csv): {path}")


class CYPMDataset(torch.utils.data.Dataset):
    """
    Fluorescence images with localization labels. Two on-disk layouts are supported (``image_format``):

    **bgy3ch** - ``BGYImg{idx}.png``: R=third structure marker, G=protein, B=nucleus.
    **bg2ch** - ``BGImg{idx}.bmp``: nucleus + protein only (ch2 usually 0), e.g. MTLS.

    After cv2.imread (BGR) and CHW conversion the channels map as:
      channel 0 = B file = nucleus (structure)
      channel 1 = G file = protein (target, goes through the AE)
      channel 2 = R file = microtubule/third marker (bgy3ch); usually 0 for bg2ch

    Hence no cv2.cvtColor(BGR2RGB) is needed; doing it would swap ch0/ch2 with nucleus/microtubule.

    Labels: 2-D float ``(p_cyto, p_nucleus)`` under ``labels`` or ``RealY`` in the h5.
    - deterministic (one component is 0): always use the h5 value, e.g. ``[0,1]`` / ``[1,0]``.
    - partial (both > 0, usually ``[0.5,0.5]`` in the training h5) is handled by ``partial_label_mode``:
      ``h5`` - always use h5 (val/test default);
      ``random`` - draw one of ``PARTIAL_LABEL_CANDIDATES`` per ``__getitem__``;
      ``pseudo`` - partial samples with ``use_pseudo`` use the fixed pseudo-label CSV/h5.
    - ``deterministic_only=True`` trains on [1,0]/[0,1] samples only.
    """
    def __init__(
        self,
        labels_h5,
        imgs_basedir,
        transform=None,
        random_partial_labels: bool = False,
        deterministic_only: bool = False,
        partial_label_mode: str = "h5",
        pseudo_labels_path: str = "",
        pseudo_split: str = "train",
        labels_key: str = "labels",
        image_format: str = "bgy3ch",
        geom_aug: bool = False,
        aug_hflip_p: float = 0.5,
        aug_vflip_p: float = 0.5,
        aug_translate_px: int = 16,
    ):
        import h5py

        self.imgbasedir = imgs_basedir
        self.transform = transform
        self.image_format = (image_format or "bgy3ch").strip().lower()
        if self.image_format not in ("bgy3ch", "bg2ch"):
            raise ValueError(
                f"image_format must be bgy3ch/bg2ch, got {image_format!r}"
            )
        self.geom_aug = bool(geom_aug)
        self.aug_hflip_p = float(aug_hflip_p)
        self.aug_vflip_p = float(aug_vflip_p)
        self.aug_translate_px = int(max(0, aug_translate_px))
        self.deterministic_only = bool(deterministic_only)
        mode = (partial_label_mode or "h5").strip().lower()
        if random_partial_labels and mode == "h5":
            mode = "random"
        if mode not in ("h5", "random", "pseudo"):
            raise ValueError(
                f"partial_label_mode must be h5/random/pseudo, got {partial_label_mode!r}"
            )
        self.partial_label_mode = mode

        key = (labels_key or "labels").strip()
        with h5py.File(labels_h5, 'r') as f:
            if key in f:
                self.labels = f[key][()]
            elif "labels" in f:
                self.labels = f["labels"][()]
            elif "RealY" in f:
                self.labels = f["RealY"][()]
            else:
                raise KeyError(
                    f"label dataset {key!r} not found in {labels_h5} (no labels/RealY either)"
                )

        self.partial_indices = []
        self.deterministic_indices = []
        for idx in range(len(self.labels)):
            label = self.labels[idx].astype(np.float32)
            if bool((label > 0).all()):
                self.partial_indices.append(idx)
            else:
                self.deterministic_indices.append(idx)
        self.partial_indices = np.array(self.partial_indices, dtype=np.int64)
        self.deterministic_indices = np.array(self.deterministic_indices, dtype=np.int64)
        self._partial_index_set = set(self.partial_indices.tolist())

        self._pseudo_cyt = None
        self._pseudo_nuc = None
        self._pseudo_use = None
        self.n_pseudo_partial = 0
        self.n_skipped_no_pseudo = 0
        skip_mask = np.zeros(len(self.labels), dtype=bool)
        if self.partial_label_mode == "pseudo":
            if not pseudo_labels_path or not os.path.isfile(pseudo_labels_path):
                raise FileNotFoundError(
                    f"partial_label_mode=pseudo needs a pseudo-label file, none found: {pseudo_labels_path!r}"
                )
            store = _load_pseudo_label_store(
                pseudo_labels_path, pseudo_split, len(self.labels)
            )
            self._pseudo_cyt = store["p_cyt"]
            self._pseudo_nuc = store["p_nuc"]
            self._pseudo_use = store["use_pseudo"]
            self.n_pseudo_partial = int(
                np.sum(store["is_partial"] & store["use_pseudo"])
            )
            skip_mask = store["is_partial"] & ~store["use_pseudo"]
            self.n_skipped_no_pseudo = int(skip_mask.sum())

        if self.deterministic_only:
            self.active_indices = self.deterministic_indices.copy()
        else:
            self.active_indices = np.nonzero(~skip_mask)[0].astype(np.int64)

    def _label_for_index(self, index: int) -> np.ndarray:
        """Return the (2,) float32 label vector."""
        if index not in self._partial_index_set:
            return self.labels[index].astype(np.float32)
        if self.partial_label_mode == "random":
            k = int(np.random.randint(0, len(PARTIAL_LABEL_CANDIDATES)))
            return PARTIAL_LABEL_CANDIDATES[k].copy()
        if self.partial_label_mode == "pseudo":
            if self._pseudo_use is not None and bool(self._pseudo_use[index]):
                return np.array(
                    [self._pseudo_cyt[index], self._pseudo_nuc[index]],
                    dtype=np.float32,
                )
            raise RuntimeError(
                f"partial sample idx={index} has no pseudo-label; it should have been filtered by active_indices"
            )
        return self.labels[index].astype(np.float32)

    def __len__(self):
        return len(self.active_indices)

    def _geom_augment_chw(self, img: np.ndarray) -> np.ndarray:
        """Apply the same flip/translation to the whole CHW image so target and structure stay aligned."""
        out = img
        if self.aug_hflip_p > 0 and np.random.rand() < self.aug_hflip_p:
            out = np.ascontiguousarray(out[:, :, ::-1])
        if self.aug_vflip_p > 0 and np.random.rand() < self.aug_vflip_p:
            out = np.ascontiguousarray(out[:, ::-1, :])
        tmax = self.aug_translate_px
        if tmax > 0:
            dy = int(np.random.randint(-tmax, tmax + 1))
            dx = int(np.random.randint(-tmax, tmax + 1))
            if dy != 0 or dx != 0:
                # Warp in HWC with border reflect, then back to CHW
                hwc = np.transpose(out, (1, 2, 0))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                h, w = hwc.shape[:2]
                hwc = cv2.warpAffine(
                    hwc,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                if hwc.ndim == 2:
                    hwc = hwc[:, :, None]
                out = np.ascontiguousarray(hwc.transpose(2, 0, 1))
        return out

    def __getitem__(self, index):
        sample_idx = int(self.active_indices[index])
        img = read_fluorescence_sample_chw(
            self.imgbasedir, sample_idx, self.image_format
        )
        if self.geom_aug:
            img = self._geom_augment_chw(img)

        label = self._label_for_index(sample_idx)
        y_final = torch.from_numpy(label)
        img_tensor = torch.from_numpy(img)
        confidence_tensor = torch.zeros(0, dtype=torch.float32)
        index_tensor = torch.tensor(sample_idx, dtype=torch.long)

        # Partial-label learning: return whether the sample has dual labels plus its candidate set
        is_partial = sample_idx in self._partial_index_set
        if is_partial and self.partial_label_mode == "random":
            candidates = torch.from_numpy(PARTIAL_LABEL_CANDIDATES.copy())
        else:
            # Single-label sample: all-zero candidates of the same shape to keep batches rectangular
            candidates = torch.zeros((3, 2), dtype=torch.float32)

        return img_tensor, y_final, confidence_tensor, index_tensor, is_partial, candidates


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.; self.avg = 0.; self.sum = 0.; self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def struct_two_channel(img: torch.Tensor, ch_indices) -> torch.Tensor:
    """img: (B,3,H,W) -> (B,2,H,W). Default [0,2] = nucleus + microtubule (excludes ch1 protein)."""
    i0, i1 = int(ch_indices[0]), int(ch_indices[1])
    return img[:, [i0, i1], :, :].contiguous()


# Min-SNR loss weighting (Hang et al. 2023): weight eps-MSE by per-sample SNR so large t (low SNR)
# no longer dominates. For eps prediction: weight(t) = min(SNR(t), gamma) / SNR(t); gamma=5 is common.
# stratified-t: split [0, T) into B bins, draw one t per bin, then shuffle. Each batch covers the
# whole t axis, cutting the variance of logged losses without changing the objective.

def minsnr_weight(diffusion, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """Min-SNR sample weight for eps prediction; returns (B,) float32 with mean ~1 (exactly 1 for gamma=inf).

    alphas_cumprod comes from SpacedDiffusion; t must index [0, diffusion.num_timesteps).
    """
    device = t.device
    alpha_bar = torch.as_tensor(
        diffusion.alphas_cumprod, device=device, dtype=torch.float32
    )[t]
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    w = torch.clamp(snr, max=float(gamma)) / snr.clamp_min(1e-8)
    return w.float()


def stratified_t_sample(num_timesteps: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Split [0, T) into batch_size equal bins, draw one t per bin, then shuffle.
    Falls back to randint when batch_size > num_timesteps."""
    T = int(num_timesteps)
    B = int(batch_size)
    if B <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if T <= B:
        return torch.randint(0, T, (B,), device=device)
    bin_w = T // B
    base = torch.arange(B, device=device) * bin_w
    offset = torch.randint(0, bin_w, (B,), device=device)
    t = (base + offset).clamp_max(T - 1)
    perm = torch.randperm(B, device=device)
    return t[perm]


def apply_cfg_train_dropout(
    z_struct: torch.Tensor,
    y: torch.Tensor,
    cfg_train_dprob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-time CFG: zero z_struct and y together with probability p (learns the unconditional branch)."""
    p = float(cfg_train_dprob)
    if p <= 0.0:
        return z_struct, y
    drop = torch.rand(z_struct.shape[0], device=z_struct.device) < p
    if not drop.any():
        return z_struct, y
    z_out = z_struct.clone()
    y_out = y.clone()
    z_out[drop] = 0.0
    y_out[drop] = 0.0
    return z_out, y_out


def _parse_index_list(s: str) -> list[int]:
    return [int(x.strip()) for x in (s or "").split(",") if x.strip()]


def expand_vis_indices_sequential(
    base_indices: list[int],
    vis_round: int,
    dataset_len: int,
    *,
    enabled: bool = True,
    max_total: int = 0,
) -> list[int]:
    """Append one extra sample per visualization round to the fixed indices (round 1 base only, round 2 +1, ...)."""
    if not base_indices:
        return []
    out = list(base_indices)
    if enabled and int(vis_round) > 1:
        extra_start = max(base_indices) + 1
        n_extra = int(vis_round) - 1
        out.extend(extra_start + k for k in range(n_extra) if extra_start + k < dataset_len)
    if int(max_total) > 0:
        out = out[: int(max_total)]
    return out


def _init_ema_params(model: torch.nn.Module) -> dict:
    """Track requires_grad parameters only (matches the optimizer)."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _ema_update(ema_params: dict, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                ema_params[n].mul_(decay).add_(p.data, alpha=1.0 - decay)


def _ema_state_dict(model: torch.nn.Module, ema_params: dict) -> dict:
    out = {k: v.clone() for k, v in model.state_dict().items()}
    for n, v in ema_params.items():
        out[n] = v.clone()
    return out


def diffusion_training_loss_eps(
    diffusion,
    model,
    x_start: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: dict,
    noise=None,
):
    """Simple epsilon prediction loss: mean((eps - eps_theta)^2).

    Matches GaussianDiffusion.training_losses with learn_sigma=False.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    x_t = diffusion.q_sample(x_start, t, noise=noise)
    model_out = model(x_t, t, **model_kwargs)
    loss_eps = mean_flat((noise - model_out) ** 2)
    return loss_eps


def diffusion_training_loss_x_start(
    diffusion,
    model,
    x_start: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: dict,
    noise=None,
):
    """Plain MSE under model_mean_type=START_X: mean((x0 - x0_hat)^2).

    Matches the loss term of GaussianDiffusion.training_losses with LossType.MSE and learn_sigma=False.
    """
    terms = diffusion.training_losses(
        model, x_start, t, model_kwargs=model_kwargs, noise=noise
    )
    return terms["loss"]


def _pred_channels_from_model(model) -> int:
    return int(getattr(getattr(model, "core", model), "pred_channels", 4))


def diffusion_training_loss_flow(
    diffusion,
    model,
    x_start: torch.Tensor,
    model_kwargs: dict,
    noise=None,
    return_layer=None,
):
    """Flow matching velocity loss: mean((v - v_theta)^2).

    - t is drawn from a logit-normal (diffusion.sample_t)
    - forward: x_t = (1 - t) * x_0 + t * eps
    - target v = eps - x_0
    - recovered x0_hat = x_t - t*v_theta (used by recon L1 etc.)

    Returns (loss_v, h_student, x0_hat, t); h_student is set only when return_layer is not None.
    """
    t = diffusion.sample_t(x_start.shape[0], device=x_start.device)
    if noise is None:
        noise = torch.randn_like(x_start)
    x_t = diffusion.q_sample(x_start, t, noise=noise)
    t_scaled = t * 1000.0
    h_student = None
    if return_layer is not None:
        model_out, h_student = model(
            x_t, t_scaled, return_layer=return_layer, **model_kwargs,
        )
    else:
        model_out = model(x_t, t_scaled, **model_kwargs)
    pc = _pred_channels_from_model(model)
    v_pred = model_out[:, :pc]
    target = noise - x_start
    loss_v = mean_flat((target - v_pred) ** 2)
    t_ = t.view(-1, *([1] * (x_start.ndim - 1)))
    x0_hat = x_t[:, :pc] - t_ * v_pred
    return loss_v, h_student, x0_hat, t


def flow_latent_moment_loss(
    x0_hat: torch.Tensor,
    x0_gt: torch.Tensor,
    t_flow: torch.Tensor,
    std_only: bool = False,
) -> tuple[torch.Tensor, float, float]:
    """Flow path: align per-channel spatial mean/std of x0_hat and x0 with L1, weighted by (1-t).
    x0_hat / x0_gt: (B, C, H, W); t_flow: (B,) continuous t in [0,1].
    Returns (loss, w_mean_component, w_std_component) for backward and logging.
    """
    _eps = 1e-6
    gt_mean = x0_gt.mean(dim=(2, 3))
    gt_std = x0_gt.std(dim=(2, 3), unbiased=False).clamp(min=_eps)
    pr_mean = x0_hat.mean(dim=(2, 3))
    pr_std = x0_hat.std(dim=(2, 3), unbiased=False).clamp(min=_eps)
    loss_mean_per = (pr_mean - gt_mean).abs().mean(dim=1)
    loss_std_per = (pr_std - gt_std).abs().mean(dim=1)
    w = (1.0 - t_flow).clamp(min=0.0)
    w_mean = (w * loss_mean_per).mean()
    w_std = (w * loss_std_per).mean()
    if std_only:
        loss = w_std
    else:
        loss = w_mean + w_std
    return loss, float(w_mean.detach().item()), float(w_std.detach().item())


def ratio_exclusive_soft_from_image(
    protein_img: torch.Tensor,
    struct_img: torch.Tensor,
    struct_ch_indices,
    mask_power: float = 2.0,
    mask_temp: float = 1.5,
    cell_thr: float = 0.05,
    nuc_thr: float = 0.10,
) -> torch.Tensor:
    """Estimate [p_cyto, p_nucleus] with DAPI hard masks (gradients flow into protein_img).

    Steps:
      1. cell_mask = (DAPI_01 + ER_01) > cell_thr  <- real cell area (excludes background -1->0)
      2. nuc_mask  = (DAPI_01 > nuc_thr) & cell_mask <- bright DAPI = nucleus
      3. cyto_mask = cell_mask - nuc_mask            <- non-nuclear cell area = cytoplasm
      4. Integrate protein intensity over the nucleus/cytoplasm masks and divide by the total.

    Compared with the old exclusive soft mask:
      - fixes the bug where cell_mask used abs().sum() > 0.05, letting background (-1 -> abs=1) cover everything
      - uses DAPI (ch0) to delimit the nucleus, removing ER (ch2) bleed-through inside it
      - measured MAE is ~43% lower than the old algorithm

    Args:
      protein_img : (B,1,H,W) in [-1,1]; gradients flow through this tensor
      struct_img  : (B,3,H,W) in BGY order (ch0=DAPI nucleus, ch1=protein, ch2=ER/microtubule)
      struct_ch_indices : [c0, c1] = [nucleus channel, ER channel] (usually [0, 2])
      nuc_thr     : DAPI [0,1] threshold for the nucleus (default 0.10)
      cell_thr    : (DAPI+ER) sum threshold below which pixels are background (default 0.05)
    Returns: (B,2) = [cyto_share, nuc_share]
    """
    c0 = int(struct_ch_indices[0])   # DAPI/nucleus channel index
    eps = 1e-8

    nuc_ch  = (struct_img[:, c0 : c0 + 1].clamp(-1.0, 1.0) + 1.0) * 0.5   # DAPI [0,1]
    # struct_ch_indices[1] is the ER/microtubule channel
    c1 = int(struct_ch_indices[1])
    er_ch   = (struct_img[:, c1 : c1 + 1].clamp(-1.0, 1.0) + 1.0) * 0.5   # ER  [0,1]
    prot    = (protein_img.clamp(-1.0, 1.0) + 1.0) * 0.5                   # protein [0,1]

    # 1. Cell area: DAPI + ER above the background threshold (background normalizes to 0)
    cell_mask = ((nuc_ch + er_ch) > float(cell_thr)).float()

    # 2. Nucleus: DAPI above nuc_thr inside the cell
    nuc_mask  = (nuc_ch > float(nuc_thr)).float() * cell_mask

    # 3. Cytoplasm: non-nuclear cell area (ER signal inside the nucleus is excluded)
    cyto_mask = (1.0 - nuc_mask) * cell_mask

    # 4. Integrate to get the shares (gradients flow through prot only)
    total      = (prot * cell_mask).sum(dim=(2, 3)).clamp_min(eps)
    cyto_share = (prot * cyto_mask).sum(dim=(2, 3)) / total
    nuc_share  = (prot * nuc_mask ).sum(dim=(2, 3)) / total
    return torch.cat([cyto_share, nuc_share], dim=1)


def ratio_mae_from_x0_hat_chunked(
    ae,
    x0_hat: torch.Tensor,
    scale_factor: float,
    struct_img: torch.Tensor,
    label: torch.Tensor,
    struct_ch_indices,
    *,
    mask_power: float = 2.0,
    mask_temp: float = 1.5,
    cell_thr: float = 0.05,
    nuc_thr: float = 0.10,
    decode_chunk: int = 0,
) -> torch.Tensor:
    """Decode x0_hat and compute the ratio MAE; chunked decode lowers peak VRAM.

    Returns per-sample MAE of shape (B,).
    """
    bsz = int(x0_hat.shape[0])
    chunk = int(decode_chunk or 0)
    if chunk <= 0 or chunk >= bsz:
        x_recon = decode_protein_latent_train(ae, x0_hat, scale_factor)
        y_ratio = ratio_exclusive_soft_from_image(
            x_recon,
            struct_img,
            struct_ch_indices,
            cell_thr=cell_thr,
            nuc_thr=nuc_thr,
        )
        return (y_ratio - label).abs().mean(dim=1)

    mae_parts = []
    for s in range(0, bsz, chunk):
        e = min(bsz, s + chunk)
        x_recon_s = decode_protein_latent_train(ae, x0_hat[s:e], scale_factor)
        y_ratio_s = ratio_exclusive_soft_from_image(
            x_recon_s,
            struct_img[s:e],
            struct_ch_indices,
            cell_thr=cell_thr,
            nuc_thr=nuc_thr,
        )
        mae_parts.append((y_ratio_s - label[s:e]).abs().mean(dim=1))
    return torch.cat(mae_parts, dim=0)


# Latent scale (ProLDM scale_by_std): the DiT expects inputs near N(0, I), while AE latents have std != 1.
#   scale_factor = 1 / z.flatten().std()
# Training and sampling use the scaled z; AE.decode requires dividing it back.
# ProLDM estimates from the first batch only; here we accumulate sum / sum_sq over the full training set
# and cache the result in the AE and experiment dirs.

def encode_protein_latent(ae, protein_img: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """AE-encode then multiply by scale_factor, returning a ~N(0,1) latent (no gradient through the AE)."""
    with torch.no_grad():
        z = ae.encode(protein_img).sample()
    return z * scale_factor


def _repa_teacher_input(img: torch.Tensor, args) -> torch.Tensor:
    """Build the SubCell REPA input (do not simply slice the first N img channels).
    bgy3ch semantics here: ch0=nucleus (DNA), ch1=protein, ch2=third structure (usually ER).
    Official SubCell rybg order: [MT, ER, DNA, Protein].
    With 3 channels the teacher receives [ER, DNA, Protein] and inserts MT=0 internally.
    """
    nch = int(getattr(args, "subcell_channels", 3))
    prot_ch = int(getattr(args, "protein_ch_index", 1) or 1)
    sch = list(getattr(args, "struct_ch_indices", [0, 2]) or [0, 2])
    if len(sch) < 2:
        raise ValueError("--struct-ch-indices needs two indices [nucleus, ER/third marker]")
    dna_ch, er_ch = int(sch[0]), int(sch[1])

    if nch == 1:
        return protein_channel(img, prot_ch)

    if img.shape[1] <= max(prot_ch, dna_ch, er_ch):
        raise ValueError(
            f"SubCell needs channel indices protein={prot_ch}, DNA={dna_ch}, ER={er_ch},"
            f"but the batch has only {img.shape[1]} channels"
        )

    protein = img[:, prot_ch : prot_ch + 1]
    er = img[:, er_ch : er_ch + 1]
    dna = img[:, dna_ch : dna_ch + 1]

    if nch == 3:
        # [ER, DNA, Protein] -> the teacher inserts MT=0 -> [0, ER, DNA, Protein]
        return torch.cat([er, dna, protein], dim=1)

    if nch == 4:
        # [MT=0, ER, DNA, Protein] (MT zeroed when there is no separate microtubule channel)
        mt = torch.zeros_like(protein)
        return torch.cat([mt, er, dna, protein], dim=1)

    raise ValueError(f"SubCell supports 1/3/4 channels, got subcell_channels={nch}")


def decode_protein_latent_train(ae, z_scaled: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Training decode: gradients flow back into z_scaled (the input latent is differentiable even with a frozen AE).
    Same numeric path as decode_protein_latent: divide by scale_factor, then ae.decode.
    """
    z_unscaled = z_scaled / float(scale_factor)
    decoded = ae.decode(z_unscaled)
    if hasattr(decoded, "sample") and callable(decoded.sample):
        return decoded.sample()
    if hasattr(decoded, "sample") and not callable(getattr(decoded, "sample", None)):
        return decoded.sample
    return decoded


@torch.no_grad()
def decode_protein_latent(ae, z: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """DiT output z -> divide by scale_factor to restore the AE scale -> decode to an image."""
    return decode_protein_latent_train(ae, z, scale_factor)


@torch.no_grad()
def compute_scale_factor_full(
    ae, dataset, device, batch_size, num_workers, logger, protein_ch_index: int = 1
) -> dict:
    """Encode the whole training set once and estimate z mean/std from running sum / sum_sq.
    Returns a dict with scale_factor, mean, std, n_elements and n_images.
    """
    tmp_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    was_training = ae.training
    ae.eval()

    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    n_images = 0
    t0 = time.time()
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    logger.info(
        f"[scale_by_std] encoding full training set ({len(dataset)} images, ~{n_batches} batches, "
        f"bs={batch_size}) to estimate z.std ... (running even without progress logs)"
    )
    for bi, batch in enumerate(tmp_loader):
        img = batch[0].to(device, non_blocking=True)
        protein_img = protein_channel(img, protein_ch_index)
        z = ae.encode(protein_img).sample().float()
        total_sum += z.sum().item()
        total_sum_sq += (z * z).sum().item()
        total_count += z.numel()
        n_images += z.shape[0]
        if (bi + 1) % max(1, n_batches // 10) == 0 or (bi + 1) == n_batches:
            logger.info(
                f"[scale_by_std] progress {bi + 1}/{n_batches} batches, "
                f"{n_images}/{len(dataset)} images, {time.time() - t0:.0f}s elapsed"
            )

    mean = total_sum / max(total_count, 1)
    var = total_sum_sq / max(total_count, 1) - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    scale_factor = 1.0 / max(std, 1e-8)

    if was_training:
        ae.train()

    logger.info(
        f"[scale_by_std] done in {time.time() - t0:.1f}s over {n_images} images "
        f"({total_count} latent elements): z.mean={mean:.4f}, z.std={std:.4f} "
        f"→ scale_factor={scale_factor:.4f}"
    )
    return {
        "scale_factor": scale_factor,
        "mean": mean,
        "std": std,
        "n_elements": total_count,
        "n_images": n_images,
    }


@torch.no_grad()
def compute_struct_scale_factor_full(
    struct_encoder, dataset, struct_ch_indices, device, batch_size, num_workers, logger,
) -> dict:
    """Estimate the element-wise std of struct_encoder outputs over the full training set; struct_scale_factor = 1/std."""
    tmp_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    was_training = struct_encoder.training
    struct_encoder.eval()

    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    n_images = 0
    t0 = time.time()
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    logger.info(
        f"[struct_scale] encoding full training set ({len(dataset)} images, ~{n_batches} batches) "
        f"for z_struct std ... (running even without progress logs)"
    )
    for bi, batch in enumerate(tmp_loader):
        img = batch[0].to(device, non_blocking=True)
        s2 = struct_two_channel(img, struct_ch_indices)
        z = struct_encoder(s2).float()
        total_sum += z.sum().item()
        total_sum_sq += (z * z).sum().item()
        total_count += z.numel()
        n_images += z.shape[0]
        if (bi + 1) % max(1, n_batches // 10) == 0 or (bi + 1) == n_batches:
            logger.info(
                f"[struct_scale] progress {bi + 1}/{n_batches} batches, "
                f"{n_images}/{len(dataset)} images, {time.time() - t0:.0f}s elapsed"
            )

    mean = total_sum / max(total_count, 1)
    var = total_sum_sq / max(total_count, 1) - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    struct_scale_factor = 1.0 / max(std, 1e-8)

    if was_training:
        struct_encoder.train()

    logger.info(
        f"[struct_scale] done in {time.time() - t0:.1f}s over {n_images} images "
        f"({total_count} latent elements): z_struct.mean={mean:.4f}, z_struct.std={std:.4f} "
        f"→ struct_scale_factor={struct_scale_factor:.4f}"
    )
    return {
        "scale_factor": struct_scale_factor,
        "mean": mean,
        "std": std,
        "n_elements": total_count,
        "n_images": n_images,
    }


def _hpa_stats_cache_dir(main_ae_ckpt: str) -> str:
    ckpt_abs = os.path.abspath(os.path.expanduser(main_ae_ckpt))
    stem = os.path.splitext(os.path.basename(ckpt_abs))[0].replace(".", "_")
    root = os.path.join(os.path.expanduser("~"), ".cache", "dit-main", "hpa_ae", stem)
    os.makedirs(root, exist_ok=True)
    return root


def _scale_cache_filename(main_ae_ckpt: str, dataset_root: str) -> str:
    tag = os.path.basename(os.path.normpath(dataset_root)) or "data"
    return os.path.join(_hpa_stats_cache_dir(main_ae_ckpt), f"scale_factor_{tag}_hpa.txt")


def _struct_scale_cache_filename(main_ae_ckpt: str, dataset_root: str, struct_backend: str) -> str:
    """Cache path for the struct latent scale."""
    tag = os.path.basename(os.path.normpath(dataset_root)) or "data"
    sb = (struct_backend or "cnn").strip()
    if sb == "hpa_vae":
        root = _hpa_stats_cache_dir(main_ae_ckpt)
        return os.path.join(root, f"struct_scale_factor_{tag}_hpa.txt")
    root = os.path.join(os.path.expanduser("~"), ".cache", "dit-main", "struct_cnn")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"struct_scale_factor_{tag}_cnn.txt")


def _read_scale_cache(path: str):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # First non-comment line holds scale_factor
                return float(line.split()[0])
    except Exception:
        return None
    return None


def _write_scale_cache(path: str, stats: dict, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{stats['scale_factor']:.8f}\n")
        f.write(f"# z.mean={stats['mean']:.6f}\n")
        f.write(f"# z.std={stats['std']:.6f}\n")
        f.write(f"# n_images={stats['n_images']}\n")
        f.write(f"# n_elements={stats['n_elements']}\n")
        if extra:
            for k, v in extra.items():
                f.write(f"# {k}={v}\n")


def get_or_compute_scale_factor(args, ae, train_dataset, device, ae_ckpt_path, experiment_dir, logger) -> float:
    """Resolve the protein scale factor by priority:
      1. explicit --z-scale-factor (>0)
      2. cached file in the AE dir, unless --recompute-scale
      3. otherwise estimate over the full training set and write it to the AE and experiment dirs
    """
    manual = getattr(args, "z_scale_factor", 0.0) or 0.0
    if manual and manual > 0:
        logger.info(f"[scale_by_std] using manual --z-scale-factor={manual:.6f}")
        _write_scale_cache(
            os.path.join(experiment_dir, "scale_factor.txt"),
            {"scale_factor": manual, "mean": float("nan"), "std": 1.0 / manual,
             "n_images": 0, "n_elements": 0},
            extra={"source": "manual", "ae_ckpt": ae_ckpt_path},
        )
        return float(manual)

    _mckpt = getattr(args, "main_ae_ckpt_resolved", None)
    if not _mckpt:
        _mckpt = os.path.abspath(os.path.expanduser((args.main_ae_ckpt or "").strip() or DEFAULT_HPA_CKPT))
    _ds = getattr(args, "data_root_resolved", None) or _CYPM_ROOT
    cache_path = _scale_cache_filename(_mckpt, _ds)
    if not getattr(args, "recompute_scale", False):
        cached = _read_scale_cache(cache_path)
        if cached is not None and cached > 0:
            logger.info(f"[scale_by_std] loaded cache {cache_path} → scale_factor={cached:.6f}")
            _write_scale_cache(
                os.path.join(experiment_dir, "scale_factor.txt"),
                {"scale_factor": cached, "mean": float("nan"), "std": 1.0 / cached,
                 "n_images": 0, "n_elements": 0},
                extra={"source": f"cache:{cache_path}", "ae_ckpt": ae_ckpt_path},
            )
            return float(cached)

    stats = compute_scale_factor_full(
        ae, train_dataset, device,
        batch_size=args.batch_size, num_workers=args.num_workers, logger=logger,
        protein_ch_index=args.protein_ch_index,
    )
    try:
        _write_scale_cache(cache_path, stats, extra={"ae_ckpt": ae_ckpt_path, "dataset": _ds})
        logger.info(f"[scale_by_std] wrote cache → {cache_path}")
    except Exception as e:
        logger.info(f"[scale_by_std] warning: failed to write cache {cache_path}: {e}")
    _write_scale_cache(
        os.path.join(experiment_dir, "scale_factor.txt"),
        stats, extra={"source": "computed", "ae_ckpt": ae_ckpt_path, "dataset": _ds},
    )
    return float(stats["scale_factor"])


def get_or_compute_struct_scale_factor(
    args, struct_encoder, train_dataset, device, experiment_dir, logger,
) -> float:
    """Mirror of the protein branch: z_struct * struct_scale_factor is roughly N(0,1).
    Priority: --struct-z-scale-factor>0, then cache, then full-set statistics.
    """
    manual = getattr(args, "struct_z_scale_factor", 0.0) or 0.0
    if manual and manual > 0:
        logger.info(f"[struct_scale] using manual --struct-z-scale-factor={manual:.6f}")
        _write_scale_cache(
            os.path.join(experiment_dir, "struct_scale_factor.txt"),
            {"scale_factor": manual, "mean": float("nan"), "std": 1.0 / manual,
             "n_images": 0, "n_elements": 0},
            extra={"source": "manual"},
        )
        return float(manual)

    _ds = getattr(args, "data_root_resolved", None) or _CYPM_ROOT
    _sb = getattr(args, "struct_backend", "cnn")
    _mckpt = getattr(args, "main_ae_ckpt_resolved", None)
    if not _mckpt:
        _mckpt = os.path.abspath(os.path.expanduser((args.main_ae_ckpt or "").strip() or DEFAULT_HPA_CKPT))
    cache_path = _struct_scale_cache_filename(_mckpt, _ds, _sb)
    if not getattr(args, "recompute_scale", False):
        cached = _read_scale_cache(cache_path)
        if cached is not None and cached > 0:
            logger.info(f"[struct_scale] loaded cache {cache_path} → struct_scale_factor={cached:.6f}")
            _write_scale_cache(
                os.path.join(experiment_dir, "struct_scale_factor.txt"),
                {"scale_factor": cached, "mean": float("nan"), "std": 1.0 / cached,
                 "n_images": 0, "n_elements": 0},
                extra={"source": f"cache:{cache_path}"},
            )
            return float(cached)

    stats = compute_struct_scale_factor_full(
        struct_encoder, train_dataset, args.struct_ch_indices, device,
        batch_size=args.batch_size, num_workers=args.num_workers, logger=logger,
    )
    try:
        _write_scale_cache(cache_path, stats, extra={"dataset": _ds, "struct_backend": _sb})
        logger.info(f"[struct_scale] wrote cache → {cache_path}")
    except Exception as e:
        logger.info(f"[struct_scale] warning: failed to write cache {cache_path}: {e}")
    _write_scale_cache(
        os.path.join(experiment_dir, "struct_scale_factor.txt"),
        stats, extra={"source": "computed", "dataset": _ds},
    )
    return float(stats["scale_factor"])


@torch.no_grad()
def ae_sanity_check(
    ae, dataset, device, scale_factor, logger, n: int = 4, save_path: str = None,
    protein_ch_index: int = 1,
):
    """Take a few training samples and measure (a) encode->decode reconstruction error and
    (b) saturation of decode on an unscaled N(0,1) latent.
    """
    was_training = ae.training
    ae.eval()
    batch = [dataset[i] for i in range(min(n, len(dataset)))]
    img = torch.stack([b[0] for b in batch], dim=0).to(device)
    protein = protein_channel(img, protein_ch_index)

    z = ae.encode(protein).sample()
    rec = ae.decode(z)
    rec = rec.sample() if hasattr(rec, "sample") and callable(rec.sample) else rec
    rec_l1 = (rec - protein).abs().mean().item()

    z_norm = z * scale_factor
    z_rand = torch.randn_like(z_norm)
    x_from_rand = decode_protein_latent(ae, z_rand, scale_factor)

    logger.info(
        f"[sanity] AE recon L1={rec_l1:.4f}; z_raw mean={z.mean():.3f} std={z.std():.3f}; "
        f"z*scale mean={z_norm.mean():.3f} std={z_norm.std():.3f}; "
        f"decode(randn/scale) mean={x_from_rand.mean():.3f} std={x_from_rand.std():.3f} "
        f"min={x_from_rand.min():.3f} max={x_from_rand.max():.3f}"
    )

    if save_path is not None:
        try:
            d = os.path.dirname(save_path) or "."
            os.makedirs(d, exist_ok=True)
            _dummy_elapsed = 0.0
            save_combined_image(protein, rec, epoch=-1, elapsed_time=_dummy_elapsed, save_dir=d)
            # save_combined_image names files f"epoch_{epoch:04d}", so -1 becomes -0001
            tmp = os.path.join(d, f"epoch_{-1:04d}_combined.png")
            if os.path.isfile(tmp) and tmp != save_path:
                os.replace(tmp, save_path)
        except Exception as e:
            logger.info(f"[sanity] warn: save recon image failed: {e}")
    if was_training:
        ae.train()


def save_combined_image(
    realimage, x_samples_ddim, epoch, elapsed_time, save_dir="", split_label: str = "val",
):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = len(realimage)
    combined_pairs = []

    realimage = realimage.permute(0, 2, 3, 1).cpu().detach().numpy()
    x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).cpu().detach().numpy()

    for n in range(batch_size):
        real_img = realimage[n, :, :, 0]
        real_img = np.clip((real_img + 1) * 127.5, 0, 255).astype(np.uint8)
        real_img_green = np.zeros((real_img.shape[0], real_img.shape[1], 3), dtype=np.uint8)
        real_img_green[:, :, 1] = real_img

        gen_img = x_samples_ddim[n, :, :, 0]
        gen_img = np.clip((gen_img + 1) * 127.5, 0, 255).astype(np.uint8)
        gen_img_green = np.zeros((gen_img.shape[0], gen_img.shape[1], 3), dtype=np.uint8)
        gen_img_green[:, :, 1] = gen_img

        pair_img = np.vstack((real_img_green, gen_img_green))
        combined_pairs.append(pair_img)

    combined_image = np.hstack(combined_pairs)
    t_io0 = time.time()
    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_combined.png")
    cv2.imwrite(save_path, combined_image)
    io_time = time.time() - t_io0
    per_image_gen = elapsed_time / max(batch_size, 1)
    print(f"  [{split_label}] Saved combined image: {save_path}")
    print(
        f"  [{split_label}] DDIM wall time: {elapsed_time:.2f}s ({per_image_gen:.2f}s / image), "
        f"save png: {io_time:.3f}s"
    )


def _patch_label_embedder_mlp_only(model) -> callable:
    """DDIM visualization: use the label MLP directly, without endpoint embedding interpolation."""
    le = model.label_embedder
    _orig = le.forward
    le.forward = lambda labels: le._mlp_embed(labels.float())

    def restore():
        le.forward = _orig

    return restore


def _oracle_labels_from_indices(dataset, index_tensor: torch.Tensor) -> torch.Tensor:
    """Read oracle (p_cyto, p_nucleus) from the h5 labels; visualization never uses pseudo-labels."""
    labels = []
    for idx in index_tensor.tolist():
        labels.append(
            torch.from_numpy(dataset.labels[int(idx)].astype(np.float32))
        )
    return torch.stack(labels, dim=0)


def _collate_train_vis_with_oracle_labels(train_dataset, indices: list[int]):
    """Take images from train_dataset but always use h5 oracle labels (visualization shows ground truth even with pseudo-label training)."""
    rows = []
    for i in indices:
        sample = train_dataset[i]
        img, _, conf, idx_t = sample[0], sample[1], sample[2], sample[3]
        sample_idx = int(idx_t.item())
        y = torch.from_numpy(train_dataset.labels[sample_idx].astype(np.float32))
        rows.append((img, y, conf, idx_t))
    return default_collate(rows)


@torch.no_grad()
def val_(val_loader, train_dataset, vis_dataset, model, ae, struct_encoder, diffusion_train, diffusion_ddim,
         epoch, args, device, logger, vis_dir, scale_factor, struct_scale_factor: float,
         ema_params=None):
    """diffusion_train uses the full step count for the same eps-MSE as training (plus the optional weighted L1 set by --recon-l1-space); diffusion_ddim uses the DDIM step count.

    Notes:
      - Validation loss still runs on val_loader; DDIM visualization takes fixed indices from vis_dataset (test or val per --vis-split, built in main), falling back to train_dataset when empty.
      - DDIM visualization always uses h5 oracle labels and direct model sampling (no image interpolation, no label_emb endpoint interpolation).
      - When ema_params is not None, the trainable DiT parameters are temporarily swapped for EMA and restored afterwards (struct_encoder always uses online weights).
      - Protein latents are already scaled by scale_factor and struct latents by struct_scale_factor, so training/validation happen near N(0,1);
        after DDIM sampling only the protein latent is divided by scale_factor before decode.
      - With cfg_scale>1: the batch is doubled with identical noise for both branches; flow mixes v inside the ODE, DDPM mixes eps/x0 via CFGModelWrapper.
    """
    model.eval()
    struct_encoder.eval()
    Loss = AverageMeter()
    sum_iter = len(val_loader)

    run_ddim = (epoch + 1) % args.val_every == 0

    max_val_batches = getattr(args, "val_max_batches", None)

    _ema_backup = None
    if ema_params is not None:
        _ema_backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                _ema_backup[n] = p.data.clone()
                p.data.copy_(ema_params[n])
    try:
        for iter_idx, batch in enumerate(val_loader):
            if max_val_batches is not None and max_val_batches > 0 and iter_idx >= max_val_batches:
                break
            # CYPMDataset returns 6-tuples (with is_partial, candidates); older 4-tuples still work
            if len(batch) >= 4:
                img, label, confidence, batch_indices = batch[:4]
            else:
                raise ValueError(f"unexpected number of val batch fields: {len(batch)}")
            img = img.to(device)
            label = label.to(device)
            confidence = confidence.to(device)

            # cv2 reads BGR: ch0=nucleus (B), ch1=protein (G), ch2=microtubule (R)
            protein_img = protein_channel(img, args.protein_ch_index)  # (B, 1, 256, 256)
            realimage = protein_img

            z_protein = encode_protein_latent(ae, protein_img, scale_factor)

            s2 = struct_two_channel(img, args.struct_ch_indices)
            z_struct = struct_encoder(s2) * struct_scale_factor

            model_kwargs = dict(z_struct=z_struct, y=label)
            _diff_type = getattr(args, "diffusion_type", "ddpm")
            if _diff_type == "flow":
                loss_v, _, _, _ = diffusion_training_loss_flow(
                    diffusion_train, model, z_protein, model_kwargs,
                )
                loss = loss_v.mean()
            else:
                if getattr(args, "stratified_t", True):
                    t = stratified_t_sample(diffusion_train.num_timesteps, z_protein.shape[0], device)
                else:
                    t = torch.randint(0, diffusion_train.num_timesteps, (z_protein.shape[0],), device=device)
                if diffusion_train.model_mean_type == ModelMeanType.START_X:
                    loss_x0 = diffusion_training_loss_x_start(
                        diffusion_train, model, z_protein, t, model_kwargs,
                    )
                    loss = loss_x0.mean()
                else:
                    loss_eps = diffusion_training_loss_eps(
                        diffusion_train, model, z_protein, t, model_kwargs,
                    )
                    _gamma = float(getattr(args, "minsnr_gamma", 0.0))
                    if _gamma > 0:
                        w = minsnr_weight(diffusion_train, t, gamma=_gamma)
                        loss = (w * loss_eps).mean()
                    else:
                        loss = loss_eps.mean()
            Loss.update(loss.item(), n=len(img))

            if run_ddim and iter_idx == 0:
                # Visualization: prefer fixed test/val indices (--vis-val-indices with vis_dataset from main), else the train set
                _vis_bs = max(int(args.n_samples), int(args.val_batch_size), 1)
                vis_val_s = (getattr(args, "vis_val_indices", "") or "").strip()
                vis_idx_s = (getattr(args, "vis_train_indices", "") or "").strip()
                _holdout_name = getattr(args, "vis_split_resolved", "val")
                _vis_indices_resolved: list[int] | None = None
                if vis_val_s:
                    base_indices = _parse_index_list(vis_val_s)
                    if not base_indices:
                        raise ValueError(
                            "--vis-val-indices parsed to empty (expected comma-separated ints, e.g. 0,1,2,3)"
                        )
                    _n_hold = len(vis_dataset)
                    _vis_round = max(1, (int(epoch) + 1) // max(int(args.val_every), 1))
                    _seq_extra = bool(getattr(args, "vis_seq_extra", True))
                    _seq_max = int(getattr(args, "vis_seq_extra_max", 0) or 0)
                    indices = expand_vis_indices_sequential(
                        base_indices,
                        _vis_round,
                        _n_hold,
                        enabled=_seq_extra,
                        max_total=_seq_max,
                    )
                    for i in base_indices:
                        if i < 0 or i >= _n_hold:
                            raise ValueError(
                                f"--vis-val-indices contains out-of-range index {i} "
                                f"(vis_split={_holdout_name}, length {_n_hold})"
                            )
                    if _seq_extra and _vis_round > 1 and len(indices) == len(base_indices):
                        logger.warning(
                            f"  [vis] vis_seq_extra is on but no extra sample can be appended "
                            f"(vis_round={_vis_round}, dataset_len={_n_hold})"
                        )
                    _vis_indices_resolved = indices
                    rows = [vis_dataset[i] for i in indices]
                    _collated = default_collate(rows)
                    img_vis, label_vis = _collated[0], _collated[1]
                    _vis_split = _holdout_name
                elif vis_idx_s:
                    indices = [int(x.strip()) for x in vis_idx_s.split(",") if x.strip()]
                    if not indices:
                        raise ValueError(
                            "--vis-train-indices parsed to empty (expected comma-separated ints, e.g. 0,1,2,3)"
                        )
                    img_vis, label_vis, _, _ = _collate_train_vis_with_oracle_labels(
                        train_dataset, indices
                    )
                    _vis_split = "train"
                else:
                    _vis_ld = DataLoader(
                        train_dataset,
                        batch_size=_vis_bs,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True,
                    )
                    _vis_batch = next(iter(_vis_ld))
                    img_vis, idx_vis = _vis_batch[0], _vis_batch[3]
                    label_vis = _oracle_labels_from_indices(train_dataset, idx_vis)
                    _vis_split = "train"
                img_vis = img_vis.to(device)
                label_vis = label_vis.to(device)
                protein_vis = protein_channel(img_vis, args.protein_ch_index)
                realimage_vis = protein_vis
                z_vis = encode_protein_latent(ae, protein_vis, scale_factor)

                n_samples = img_vis.shape[0]
                if _vis_indices_resolved is None:
                    n_samples = min(int(args.n_samples), n_samples)
                y_cond = label_vis[:n_samples]

                s2_s = struct_two_channel(img_vis[:n_samples], args.struct_ch_indices)
                z_struct_s = struct_encoder(s2_s) * struct_scale_factor

                # Match the main latent grid (ProLDM->32, HPA main->64); do not hardcode 32
                _L = z_vis.shape[-1]

                _vn = int(getattr(args, "vis_noise_seed", DEFAULT_VIS_NOISE_SEED))
                if _vn >= 0:
                    _g = torch.Generator(device="cpu")
                    _g.manual_seed(_vn)
                    noise_s = torch.randn(
                        n_samples, 4, _L, _L, generator=_g, dtype=torch.float32
                    ).to(device)
                else:
                    noise_s = torch.randn(n_samples, 4, _L, _L, device=device)
                # Clamp x0 during DDIM sampling (prevents eps-solve blow-up at high t)
                _x0_clamp = float(getattr(args, "ddim_x0_clamp", 0.0) or 0.0)
                _denoised_fn = (
                    (lambda x, _c=_x0_clamp: x.clamp(-_c, _c)) if _x0_clamp > 0 else None
                )
                _cfg_scale = float(getattr(args, "cfg_scale", 1.0))
                _use_cfg = _cfg_scale > 1.0 + 1e-6
                if _use_cfg and float(getattr(args, "cfg_train_dprob", 0.0)) <= 0.0:
                    logger.warning(
                        f"  [val] cfg_scale={_cfg_scale} but cfg_train_dprob=0; "
                        "the unconditional branch may be undertrained, consider --cfg-train-dprob>0"
                    )
                from models import prepare_cfg_ddim_sampling, take_cfg_protein_latent

                _restore_label_vis = _patch_label_embedder_mlp_only(model)
                try:
                    _sample_shape, _noise_in, model_kwargs_s, sample_model, _use_cfg = (
                        prepare_cfg_ddim_sampling(
                            n_samples,
                            noise_s,
                            z_struct_s,
                            y_cond,
                            _cfg_scale,
                            _diff_type,
                            model,
                        )
                    )
                    start_time = time.time()
                    samples = diffusion_ddim.ddim_sample_loop(
                        sample_model,
                        shape=_sample_shape,
                        noise=_noise_in,
                        clip_denoised=False,
                        denoised_fn=_denoised_fn,
                        model_kwargs=model_kwargs_s,
                        progress=False,
                        device=device,
                        eta=args.ddim_eta,
                    )
                    z_gen = take_cfg_protein_latent(
                        samples, n_samples, _use_cfg, _diff_type
                    )
                    elapsed = time.time() - start_time
                    x_gen = decode_protein_latent(ae, z_gen, scale_factor)

                    save_combined_image(
                        realimage_vis[:n_samples],
                        x_gen,
                        epoch,
                        elapsed,
                        save_dir=vis_dir,
                        split_label=(
                            f"{_vis_split}{' idx=' + ','.join(map(str, _vis_indices_resolved)) if _vis_indices_resolved else ''}"
                        ),
                    )
                finally:
                    _restore_label_vis()

            # \033[K clears to end of line so the " +DDIM" suffix does not leak into the loss digits after \r
            print(
                f"\r  [val] Epoch [{epoch + 1}] [{iter_idx + 1}/{sum_iter}] loss: {Loss.avg:.4f}"
                f"{' +DDIM' if run_ddim and iter_idx == 0 else ''}\033[K",
                end="",
                flush=True,
            )

        print()
        tail = "" if run_ddim else f" (no DDIM this epoch; val_every={args.val_every})"
        if run_ddim:
            if (getattr(args, "vis_val_indices", "") or "").strip():
                _vn = getattr(args, "vis_split_resolved", "val")
                _vis_note = f"; DDIM vis={_vn} base indices {args.vis_val_indices!r}"
                if getattr(args, "vis_seq_extra", True):
                    _vr = max(1, (int(epoch) + 1) // max(int(args.val_every), 1))
                    _vis_note += f"; vis_seq_extra round={_vr}"
            elif (getattr(args, "vis_train_indices", "") or "").strip():
                _vis_note = f"; DDIM vis=fixed train indices {args.vis_train_indices!r}"
            else:
                _vis_note = "; DDIM vis=random train batch"
            _vis_ns = int(getattr(args, "vis_noise_seed", DEFAULT_VIS_NOISE_SEED))
            if _vis_ns >= 0:
                _vis_note += f"; vis_noise_seed={_vis_ns}"
            _vis_cfg = float(getattr(args, "cfg_scale", 1.0))
            if _vis_cfg > 1.0 + 1e-6:
                _vis_note += f"; cfg_scale={_vis_cfg}"
        else:
            _vis_note = ""
        if ema_params is not None:
            _vis_note += "; eval=EMA(DiT)"
        logger.info(f"  [val] Epoch [{epoch + 1}] avg loss: {Loss.avg:.4f}{tail}{_vis_note}")
    finally:
        if _ema_backup is not None:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(_ema_backup[n])
    # Do not restore model/struct_encoder train mode; the main loop sets it back explicitly each step.


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # Data root, image format and h5 naming (CYPM: CyPM; MTLS: LyMo2010*labels_.h5)
    data_root = (getattr(args, "data_root", "") or "").strip() or _load_cypm_root()
    h5tag = (getattr(args, "dataset_h5_tag", "CyPM") or "CyPM").strip()
    image_format_arg = (getattr(args, "image_format", "auto") or "auto").strip().lower()
    if image_format_arg == "auto":
        image_format = detect_image_format(data_root)
    elif image_format_arg in ("bgy3ch", "bg2ch"):
        image_format = image_format_arg
    else:
        raise ValueError(
            f"--image-format must be auto/bgy3ch/bg2ch, got {image_format_arg!r}"
        )
    args.image_format_resolved = image_format

    train_labels_h5, train_labels_key = resolve_labels_h5_path(
        data_root, "train", h5tag, required=True
    )
    val_labels_h5, val_labels_key = resolve_labels_h5_path(
        data_root, "val", h5tag, required=True
    )
    train_imgs_dir = os.path.join(data_root, "train")
    val_imgs_dir = os.path.join(data_root, "val")
    args.data_root_resolved = data_root

    if image_format == "bg2ch" and list(args.struct_ch_indices) == [0, 2]:
        args.struct_ch_indices = [0, 0]

    test_labels_h5, test_labels_key = resolve_labels_h5_path(
        data_root, "test", h5tag, required=False
    )
    test_imgs_dir = os.path.join(data_root, "test")

    for p, desc in (
        (train_labels_h5, "train labels h5"),
        (val_labels_h5, "val labels h5"),
        (train_imgs_dir, "train/"),
        (val_imgs_dir, "val/"),
    ):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Data path missing ({desc}): {p}\n"
                f"  check --data-root and --dataset-h5-tag (root={data_root!r} tag={h5tag!r})"
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    rank = 0
    world_size = 1
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    resume_raw = (getattr(args, "resume", "") or "").strip()

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if resume_raw:
            resume_abs = os.path.abspath(resume_raw)
            if not os.path.isfile(resume_abs):
                raise FileNotFoundError(
                    f"--resume checkpoint does not exist: {resume_abs}"
                )
            checkpoint_dir = os.path.dirname(resume_abs)
            experiment_dir = os.path.dirname(checkpoint_dir)
            if not os.path.isdir(experiment_dir):
                raise FileNotFoundError(
                    f"Experiment dir inferred from --resume does not exist: {experiment_dir}"
                )
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"[resume] Experiment directory: {experiment_dir}")
            logger.info(f"[resume] Will load: {resume_abs}")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(
            f"[data] root={data_root}  h5_tag={h5tag!r}  image_format={image_format}  "
            f"train_h5={os.path.basename(train_labels_h5)}({train_labels_key})  "
            f"val_h5={os.path.basename(val_labels_h5)}({val_labels_key})  "
            f"struct_ch={args.struct_ch_indices}  protein_ch={args.protein_ch_index}  "
            f"test_exists={bool(test_labels_h5) and os.path.isdir(test_imgs_dir)}"
        )
    else:
        logger = create_logger(None)

    # Main branch AE: frozen HPA VAE (CELL-Diff)
    hpa_main_ckpt = args.main_ae_ckpt_resolved
    ae = load_hpa_main_ae(hpa_main_ckpt, device)
    ae_ckpt = hpa_main_ckpt
    main_latent_size = hpa_main_latent_size(IMG_SIZE)
    logger.info(
        f"[main-ae] HPA VAE loaded {hpa_main_ckpt}  latent={main_latent_size}x{main_latent_size}x4 (frozen)"
    )

    # struct encoder backend
    # cnn    : 3 lightweight conv layers, trainable, output (B, 2, 32, 32)
    # hpa_vae: frozen CELL-Diff HPA VAE, output (B, 8, L, L);
    #          L follows the main branch by default (64 for HPA; struct also 64 or pooled to 32)
    struct_backend = getattr(args, "struct_backend", "cnn")
    if struct_backend == "hpa_vae":
        struct_freeze = True  # pretrained HPA, always frozen
        # struct latent size follows main by default; explicit override is rarely used
        _hpa_struct_L = getattr(args, "hpa_struct_latent_size", 0) or 0
        if _hpa_struct_L <= 0:
            _hpa_struct_L = main_latent_size  # 32 / 64
        struct_encoder = load_hpa_struct_encoder(
            ckpt_path=getattr(args, "hpa_vae_ckpt", "") or DEFAULT_HPA_CKPT,
            device=device,
            latent_size=_hpa_struct_L,
        )
        struct_out_ch = hpa_struct_out_channels()  # 8 (4 nucleus + 4 microtubule, concatenated)
        _struct_num_groups = 1
        _struct_latent_ch_per_group = struct_out_ch  # single-group patchify, 256 struct tokens total
        struct_spatial = _hpa_struct_L
        logger.info(
            f"Struct encoder: backend=hpa_vae ch={args.struct_ch_indices} "
            f"out=(B,{struct_out_ch},{struct_spatial},{struct_spatial}) freeze=True"
        )
    else:
        struct_freeze = getattr(args, "struct_encoder_freeze", False)
        struct_encoder = load_struct_encoder(
            args.struct_encoder_ckpt if str(args.struct_encoder_ckpt).strip() else None,
            device,
            torch.float32,
            freeze=struct_freeze,
        )
        struct_out_ch = 2
        _struct_num_groups = 1
        _struct_latent_ch_per_group = struct_out_ch  # nucleus+microtubule (2ch) patchified as one group
        struct_spatial = 32  # CNN always maps 256->32
        logger.info(
            f"Struct encoder: backend=cnn ckpt={args.struct_encoder_ckpt or '(random init)'} "
            f"ch={args.struct_ch_indices} out=(B,{struct_out_ch},{struct_spatial},{struct_spatial}) "
            f"freeze={struct_freeze} ({'no backprop' if struct_freeze else 'trained with the DiT'})"
        )

    # DiT patch_size consistency: --model fixes it (DiT-B/2 -> 2, DiT-B/4 -> 4),
    # and main_latent_size must be divisible by it (HPA main=64 usually pairs with DiT-B/4).
    # The patch size is parsed from the model name, matching DiT_models naming.
    _dit_patch_size = int(args.model.rsplit("/", 1)[-1])
    if main_latent_size % _dit_patch_size != 0:
        raise ValueError(
            f"main_latent_size={main_latent_size} is not divisible by patch={_dit_patch_size} of --model={args.model}; "
            f"recommended: HPA main + DiT-B/4 (patch=4, 64/4=16) or another divisible --model"
        )
    tokens_per_side = main_latent_size // _dit_patch_size
    n_tokens_main = tokens_per_side * tokens_per_side

    # Struct token count (after merging channels into a single patchify group).
    # --struct-tokens-per-ch N defaults to 256 (comparable to the main branch; no longer split per channel).
    # If N is not a perfect square, search the most square (patch_h, patch_w) that divides struct_spatial.
    import math as _math

    _tokens_per_ch = int(getattr(args, "struct_tokens_per_ch", 0) or 0)
    if _tokens_per_ch <= 0:
        _tokens_per_ch = 256  # 256 tokens by default (patch=4x4)

    def _find_patch_hw(spatial, target_tokens):
        """Find (ph, pw) on a spatial x spatial grid with (spatial/ph)*(spatial/pw) = target_tokens.
        Prefers the most square option (smallest |ph-pw|).
        """
        divs = [d for d in range(1, spatial + 1) if spatial % d == 0]
        candidates = []
        for ph in divs:
            gh = spatial // ph
            if target_tokens % gh != 0:
                continue
            gw = target_tokens // gh
            if gw < 1 or spatial % gw != 0:
                continue
            pw = spatial // gw
            candidates.append((abs(ph - pw), ph, pw))
        if not candidates:
            raise ValueError(
                f"Cannot obtain {target_tokens} tokens on a {spatial}x{spatial} grid with integer patches; "
                f"common valid values (spatial=64): 64,128,256 (patch 8x8/4x8/4x4)."
            )
        candidates.sort()
        return candidates[0][1], candidates[0][2]   # ph, pw

    _ph, _pw = _find_patch_hw(struct_spatial, _tokens_per_ch)
    struct_patch_size = (_ph, _pw) if _ph != _pw else _ph
    n_tokens_struct   = _tokens_per_ch  # tokens per group (for logging)

    latent_size = main_latent_size  # DiT core.input_size
    _dit_param = getattr(args, "dit_parameterization", "eps")
    _predict_xstart = _dit_param == "x0"
    _aux = bool(getattr(args, "dit_predict_aux_x0", False)) and not _predict_xstart
    _cfg_mix = "x0" if _predict_xstart else "epsilon"
    core = DiT_models[args.model](
        input_size=latent_size,
        learn_sigma=False,
        in_channels=4,
        pred_channels=4,
        predict_aux_x0=_aux,
        cfg_mix_prediction=_cfg_mix,
    )
    model = DiTWithStructSeq(
        core,
        struct_in_ch=_struct_num_groups,                   # 1 (nucleus+microtubule merged into one group)
        struct_latent_ch_per_group=_struct_latent_ch_per_group,  # 8 (hpa_vae) or 2 (cnn)
        struct_input_size=struct_spatial,
        struct_patch_size=struct_patch_size,
        label_endpoint_interp=bool(getattr(args, "label_endpoint_interp", False)),
    ).to(device)
    _n_struct_tokens_actual = n_tokens_struct * _struct_num_groups  # single group -> 256 by default
    _interp_note = ",label_endpoint_interp=on" if getattr(args, "label_endpoint_interp", False) else ""
    logger.info(
        f"[struct-inject] seq_concat: main=(latent={latent_size},patch={_dit_patch_size},"
        f"tokens={n_tokens_main}) struct=(latent={struct_spatial},patch={struct_patch_size},"
        f"tokens_per_group={n_tokens_struct},groups={_struct_num_groups},"
        f"latent_ch_per_group={_struct_latent_ch_per_group},total={_n_struct_tokens_actual}{_interp_note})"
        f" seq_len={n_tokens_main + _n_struct_tokens_actual}"
    )
    _lat_adv_early = float(getattr(args, "latent_adv_coef", 0.0) or 0.0)
    _adv_mode = str(getattr(args, "latent_adv_mode", "lddgan") or "lddgan").lower()
    if _lat_adv_early > 0 and _adv_mode == "lddgan":
        if getattr(args, "diffusion_type", "ddpm") != "ddpm":
            logger.warning(
                 "[latent-adv] LDDGAN needs the q(x_{t-1}|x_t,x_0) posterior; forcing --diffusion-type ddpm "
                 f"(was {getattr(args, 'diffusion_type', None)!r})"
            )
            args.diffusion_type = "ddpm"

    _diff_type = getattr(args, "diffusion_type", "ddpm")
    if _diff_type == "flow":
        _flow_num_steps = int(getattr(args, "flow_num_steps", 28))
        _flow_heun = bool(getattr(args, "flow_use_heun", False))
        _flow_shift = float(getattr(args, "flow_time_shift_mu", 0.0))
        _flow_sigma_min = float(getattr(args, "flow_sigma_min", 0.0))
        diffusion = create_diffusion(
            timestep_respacing="",
            learn_sigma=False,
            diffusion_type="flow",
            flow_num_steps=_flow_num_steps,
            flow_sigma_min=_flow_sigma_min,
            flow_use_heun=_flow_heun,
            flow_time_shift_mu=_flow_shift,
        )
        diffusion_ddim = create_diffusion(
            timestep_respacing=str(args.ddim_steps),
            learn_sigma=False,
            diffusion_type="flow",
            flow_num_steps=args.ddim_steps,
            flow_sigma_min=_flow_sigma_min,
            flow_use_heun=_flow_heun,
            flow_time_shift_mu=_flow_shift,
        )
    else:
        diffusion = create_diffusion(
            timestep_respacing="", learn_sigma=False, predict_xstart=_predict_xstart,
            noise_schedule=args.noise_schedule,
            diffusion_type="ddpm",
        )
        diffusion_ddim = create_diffusion(
            timestep_respacing=str(args.ddim_steps),
            learn_sigma=False,
            predict_xstart=_predict_xstart,
            noise_schedule=args.noise_schedule,
            diffusion_type="ddpm",
        )
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if _diff_type == "flow":
        logger.info(
            f"[flow] Flow Matching: num_steps={_flow_num_steps}, heun={_flow_heun}, "
            f"time_shift_mu={_flow_shift}, sigma_min={_flow_sigma_min}"
        )
        logger.info("[flow] the model predicts velocity v=eps-x0; training uses t~logit-normal, sampling Euler/Heun ODE.")
        if float(getattr(args, "minsnr_gamma", 0.0)) > 0:
            logger.info("[flow] Min-SNR does not apply to flow matching; ignoring --minsnr-gamma.")
    elif _predict_xstart:
        logger.info(
            f"[dit] dit_parameterization=x0 (predicts the clean latent directly; GaussianDiffusion START_X)"
        )
        if float(getattr(args, "minsnr_gamma", 0.0)) > 0:
            logger.info(
                "[dit] Min-SNR is designed for eps prediction only; disabled because this run predicts x0."
            )
    _cfg_dprob = float(getattr(args, "cfg_train_dprob", 0.0) or 0.0)
    _cfg_scale = float(getattr(args, "cfg_scale", 1.5))
    logger.info(
        f"[cfg] train_dropout={_cfg_dprob}, val_ddim_cfg_scale={_cfg_scale}"
        + (" (CFG enabled for validation sampling)" if _cfg_scale > 1.0 + 1e-6 else " (no CFG at validation)")
    )
    if _cfg_scale > 1.0 + 1e-6 and _cfg_dprob <= 0.0:
        logger.warning(
            "[cfg] cfg_scale>1 but cfg_train_dprob=0: the unconditional branch is untrained; try --cfg-train-dprob 0.1"
        )
    if _aux and _diff_type != "flow":
        logger.info(
            f"[dit] predict_aux_x0=True (2x4 outputs: eps + x0 heads)"
        )

    _opt_params = list(model.parameters())
    if not struct_freeze:
        _opt_params += list(struct_encoder.parameters())
        logger.info(
            f"Struct encoder params: {sum(p.numel() for p in struct_encoder.parameters()):,} (trainable)"
        )

    # REPA: load the frozen SubCell teacher and projector
    repa_teacher = None
    repa_proj = None
    _repa_coef = float(getattr(args, "repa_coef", 0.0) or 0.0)
    _repa_layer = int(getattr(args, "repa_layer", 4))
    if _repa_coef > 0:
        if _diff_type == "flow":
            logger.info("[repa] flow matching: align student intermediate-layer tokens with SubCell patch tokens.")
        from repa import REPAProjector, build_repa_teacher, repa_teacher_feature_dim

        repa_teacher = build_repa_teacher(args, device)
        repa_teacher.eval()
        _n_blocks = len(model.core.blocks)
        if _repa_layer < 0 or _repa_layer >= _n_blocks:
            raise ValueError(
                f"--repa-layer={_repa_layer} out of range (0..{_n_blocks-1} for {_n_blocks} blocks)"
            )
        _student_dim = int(model.core.hidden_size)
        _teacher_dim = int(repa_teacher_feature_dim(repa_teacher))
        repa_proj = REPAProjector(_student_dim, _teacher_dim).to(device)
        _opt_params += list(repa_proj.parameters())
        _n_student_tokens = (latent_size // _dit_patch_size) ** 2
        _n_teacher_tokens = int(repa_teacher.num_patch_tokens)
        if _n_student_tokens != _n_teacher_tokens:
            logger.info(
                f"[repa] token grid alignment: SubCell {_n_teacher_tokens} -> DiT {_n_student_tokens} "
                f"(bilinear;latent={latent_size}, patch={_dit_patch_size}, "
                f"teacher_input={repa_teacher.input_size})"
            )
        logger.info(
            f"[repa] teacher=subcell input={repa_teacher.input_size} "
            f"dtype={getattr(args, 'repa_teacher_dtype', 'fp32')} "
            f"teacher_dim={_teacher_dim}; student_layer={_repa_layer} dim={_student_dim} "
            f"tokens={_n_student_tokens}; coef={_repa_coef}; "
            f"projector_params={sum(p.numel() for p in repa_proj.parameters()):,}"
        )

    opt = torch.optim.AdamW(_opt_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))

    # LR schedule: linear warmup + CosineAnnealingLR.
    # The scheduler needs total_steps (loader length), so it is built before the training loop.
    _lr_warmup_steps = max(0, int(getattr(args, "lr_warmup_steps", 0) or 0))
    _lr_eta_min      = float(getattr(args, "lr_eta_min", 1e-5) or 1e-5)
    _lr_scheduler    = None   # initialized before the epoch loop

    latent_disc = None
    opt_d = None
    _lat_adv = float(getattr(args, "latent_adv_coef", 0.0) or 0.0)
    _adv_mode = str(getattr(args, "latent_adv_mode", "lddgan") or "lddgan").lower()
    if _lat_adv > 0:
        if _adv_mode == "lddgan":
            if _diff_type != "ddpm":
                raise ValueError(
                    "LDDGAN adversarial training needs --diffusion-type ddpm (it requires q_posterior)"
                )
            latent_disc = LatentLDDGANDiscriminator(
                protein_ch=4,
                struct_ch=int(struct_out_ch),
                spatial_size=int(latent_size),
                ndf=int(getattr(args, "latent_adv_base_ch", 64)),
                use_struct=True,
                aux_label_dim=2,
            ).to(device)
            logger.info(
                f"[latent-adv] LDDGAN transition discriminator D(x_{{t-1}},x_t,t,z_struct) (ddpm):"
                f"coef={_lat_adv}, "
                f"aux_coef={float(getattr(args, 'latent_adv_aux_coef', 10.0) or 0.0)}, "
                f"D_lr={float(getattr(args, 'latent_adv_lr', 2e-4))}, "
                f"D_params={sum(p.numel() for p in latent_disc.parameters()):,} "
                f"(with a quantitative label head for partial-label candidate selection; aux=real->y"
                f"{' + fake→0' if bool(getattr(args, 'latent_adv_aux_fake_zero', False)) else ' (no fake→0)'})"
            )
        elif _adv_mode == "plocgan":
            latent_disc = LatentPlocGANDiscriminator(
                in_ch=4 + int(struct_out_ch),
                spatial_size=int(latent_size),
                ndf=int(getattr(args, "latent_adv_base_ch", 64)),
                aux_label_dim=2,
            ).to(device)
            logger.info(
                f"[latent-adv] PLocGAN endpoint discriminator D([z_struct,x0_hat]) ({_diff_type}):"
                f"coef={_lat_adv}, "
                f"aux_coef={float(getattr(args, 'latent_adv_aux_coef', 1.0) or 0.0)}, "
                f"D_lr={float(getattr(args, 'latent_adv_lr', 2e-4))}, "
                f"D_params={sum(p.numel() for p in latent_disc.parameters()):,}"
            )
        else:
            raise ValueError(f"--latent-adv-mode supports lddgan/plocgan, got: {_adv_mode!r}")
        opt_d = torch.optim.AdamW(
            latent_disc.parameters(),
            lr=float(getattr(args, "latent_adv_lr", 2e-4)),
            betas=(0.5, 0.999),
        )

    _train_det_only = bool(getattr(args, "train_deterministic_only", False))
    _partial_mode = (getattr(args, "partial_label_mode", None) or "h5").strip().lower()
    _pseudo_path = (getattr(args, "pseudo_labels_path", "") or "").strip()
    if _train_det_only and _partial_mode == "pseudo":
        raise ValueError(
            "train_deterministic_only and partial_label_mode=pseudo cannot be combined (run stages 1 and 2 separately)"
        )
    _geom_aug = bool(getattr(args, "train_geom_aug", False))
    dataset = CYPMDataset(
        train_labels_h5,
        train_imgs_dir,
        random_partial_labels=(_partial_mode == "random"),
        deterministic_only=_train_det_only,
        partial_label_mode=_partial_mode,
        pseudo_labels_path=_pseudo_path,
        pseudo_split="train",
        labels_key=train_labels_key,
        image_format=image_format,
        geom_aug=_geom_aug,
        aug_hflip_p=float(getattr(args, "aug_hflip_p", 0.5)),
        aug_vflip_p=float(getattr(args, "aug_vflip_p", 0.5)),
        aug_translate_px=int(getattr(args, "aug_translate_px", 16)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    _ds_msg = (
        f"Dataset: {len(dataset)} images, "
        f"partial={len(dataset.partial_indices)}, "
        f"deterministic={len(dataset.deterministic_indices)}"
    )
    if _train_det_only:
        _ds_msg += " (deterministic [1,0]/[0,1] samples only)"
    elif _partial_mode == "pseudo":
        _ds_msg += (
            f"; partial_label_mode=pseudo n_pseudo={dataset.n_pseudo_partial}"
            f" skipped={dataset.n_skipped_no_pseudo}"
        )
    elif _partial_mode == "random":
        _ds_msg += f"; random_partial={PARTIAL_LABEL_CANDIDATES.tolist()}"
    if _geom_aug:
        _ds_msg += (
            f"; geom_aug=hflip({float(getattr(args, 'aug_hflip_p', 0.5)):.2f})"
            f"/vflip({float(getattr(args, 'aug_vflip_p', 0.5)):.2f})"
            f"/translate≤{int(getattr(args, 'aug_translate_px', 16))}px"
        )
    logger.info(_ds_msg)

    val_dataset = CYPMDataset(
        val_labels_h5,
        val_imgs_dir,
        labels_key=val_labels_key,
        image_format=image_format,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    _vis_split_req = (getattr(args, "vis_split", "test") or "test").strip().lower()
    if _vis_split_req not in ("val", "test"):
        raise ValueError("--vis-split must be val or test")
    if (
        _vis_split_req == "test"
        and test_labels_h5
        and os.path.isfile(test_labels_h5)
        and os.path.isdir(test_imgs_dir)
    ):
        vis_dataset = CYPMDataset(
            test_labels_h5,
            test_imgs_dir,
            labels_key=test_labels_key,
            image_format=image_format,
        )
        args.vis_split_resolved = "test"
    elif _vis_split_req == "test":
        logger.warning(
            f"[vis] --vis-split test but test data is missing ({test_labels_h5 or 'N/A'} or {test_imgs_dir}); "
            f"falling back to val/ for fixed DDIM visualization"
        )
        vis_dataset = val_dataset
        args.vis_split_resolved = "val"
    else:
        vis_dataset = val_dataset
        args.vis_split_resolved = "val"

    vis_dir = f"{experiment_dir}/samples"
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(
        f"Val dataset: {len(val_dataset)} images; "
        f"DDIM fixed visualization split={args.vis_split_resolved} "
        f"(n={len(vis_dataset)}); vis_dir: {vis_dir}"
    )

    # Latent scale factor (ProLDM scale_by_std, but estimated over the full training set)
    scale_factor = get_or_compute_scale_factor(
        args, ae, dataset, device, ae_ckpt, experiment_dir, logger,
    )
    struct_scale_factor = get_or_compute_struct_scale_factor(
        args, struct_encoder, dataset, device, experiment_dir, logger,
    )
    logger.info(
        f"[scale] protein scale_factor={scale_factor:.6f}  "
        f"struct_scale_factor={struct_scale_factor:.6f}"
    )

    # Frequency low/high band ratio: >0 manual, <=0 auto-estimated from the training power spectrum
    _recon_freq_coef_init = float(getattr(args, "recon_freq_coef", 0.0) or 0.0)
    _freq_ratio_cfg = float(getattr(args, "recon_freq_ratio", 0.3) or 0.3)
    if _recon_freq_coef_init > 0 and _freq_ratio_cfg <= 0:
        from freq_loss import estimate_low_freq_ratio_from_dataset

        _freq_space_est = str(getattr(args, "recon_freq_space", "image") or "image").lower()
        _freq_stats = estimate_low_freq_ratio_from_dataset(
            dataset,
            space=_freq_space_est,
            ae=ae,
            scale_factor=scale_factor,
            device=device,
            n_samples=int(getattr(args, "recon_freq_auto_samples", 512) or 512),
            batch_size=min(args.batch_size, 16),
            energy_fraction=float(getattr(args, "recon_freq_energy_frac", 0.85) or 0.85),
            logger=logger,
        )
        args.recon_freq_ratio = float(_freq_stats["ratio"])
        _fr_path = os.path.join(experiment_dir, "recon_freq_ratio.txt")
        with open(_fr_path, "w") as f:
            f.write(f"{args.recon_freq_ratio:.6f}\n")
            f.write(f"# space={_freq_stats['space']} n={_freq_stats['n_images']}\n")
            f.write(f"# energy_fraction={_freq_stats['energy_fraction']}\n")
            f.write(f"# HxW={_freq_stats['spatial_hw']}\n")
    elif _recon_freq_coef_init > 0:
        logger.info(f"[recon-freq] using manual ratio={_freq_ratio_cfg:.4f}")

    if getattr(args, "sanity_check", True):
        _sanity_png = os.path.join(experiment_dir, "ae_sanity_recon.png")
        ae_sanity_check(
            ae, dataset, device, scale_factor, logger, n=4, save_path=_sanity_png,
            protein_ch_index=args.protein_ch_index,
        )

    model.train()
    if not struct_freeze:
        struct_encoder.train()

    resume_ckpt = None
    train_steps = 0
    start_epoch = 0
    if resume_raw:
        resume_abs = os.path.abspath(resume_raw)
        resume_ckpt = torch.load(resume_abs, map_location="cpu", weights_only=False)
        # Older checkpoints may lack null_struct_embed; fill in the initial values
        _model_sd = resume_ckpt["model"]
        _missing = set(model.state_dict().keys()) - set(_model_sd.keys())
        if _missing:
            for k in sorted(_missing):
                if k.endswith("null_struct_embed"):
                    logger.info(f"[resume] old checkpoint lacks {k}; using random init (it will keep learning)")
                    _model_sd[k] = model.state_dict()[k].clone()
                else:
                    logger.warning(f"[resume] missing key {k}; using current model init")
                    _model_sd[k] = model.state_dict()[k].clone()
        model.load_state_dict(_model_sd, strict=True)
        opt.load_state_dict(resume_ckpt["opt"])
        if repa_proj is not None:
            _rp = resume_ckpt.get("repa_proj")
            if _rp is not None:
                _miss_r = repa_proj.load_state_dict(_rp, strict=False)
                _mkr = getattr(_miss_r, "missing_keys", [])
                _ukr = getattr(_miss_r, "unexpected_keys", [])
                if _mkr or _ukr:
                    logger.info(
                        f"[resume] repa_proj loaded non-strictly: missing={len(_mkr)} unexpected={len(_ukr)}"
                    )
                else:
                    logger.info("[resume] loaded repa_proj")
            else:
                logger.info("[resume] ckpt has no repa_proj; keeping current init (last layer zero-initialized)")

        if latent_disc is not None:
            _ld = resume_ckpt.get("latent_discriminator")
            if _ld is not None:
                if isinstance(latent_disc, LatentPlocGANDiscriminator):
                    _ld = LatentPlocGANDiscriminator.remap_legacy_state_dict(_ld)
                try:
                    _miss = latent_disc.load_state_dict(_ld, strict=False)
                    _mk = getattr(_miss, "missing_keys", [])
                    _uk = getattr(_miss, "unexpected_keys", [])
                    if _mk:
                        logger.info(
                            f"[resume] latent_discriminator loaded non-strictly: missing={len(_mk)} keys"
                        )
                    if _uk:
                        logger.info(
                            f"[resume] latent_discriminator unexpected_keys={len(_uk)}"
                        )
                    logger.info("[resume] loaded latent_discriminator")
                except Exception as _e:
                    logger.warning(
                        f"[resume] skipping latent_discriminator (expected on architecture change, e.g. plocgan->lddgan): {_e}"
                    )
                    _ld = None
            _ldo = resume_ckpt.get("latent_disc_opt")
            if _ldo is not None and opt_d is not None and _ld is not None:
                try:
                    opt_d.load_state_dict(_ldo)
                    logger.info("[resume] loaded latent_disc_opt")
                except Exception as _e:
                    logger.warning(
                        f"[resume] skipping latent_disc_opt (expected when D params differ): {_e}"
                    )
        if "struct_encoder" in resume_ckpt:
            _se_res = struct_encoder.load_state_dict(resume_ckpt["struct_encoder"], strict=False)
            _se_missing = list(getattr(_se_res, "missing_keys", []))
            _se_unexpected = list(getattr(_se_res, "unexpected_keys", []))
            if _se_missing or _se_unexpected:
                logger.warning(
                    f"[resume] struct_encoder loaded non-strictly: "
                    f"missing={len(_se_missing)} unexpected={len(_se_unexpected)}"
                )
            if _se_missing:
                raise RuntimeError(
                    "[resume] struct_encoder is incomplete in the checkpoint; "
                    f"missing={_se_missing[:5]}. Refusing to resume with a partially "
                    "random encoder (use a matching checkpoint)."
                )
        train_steps = int(resume_ckpt.get("train_steps", 0) or 0)
        start_epoch = int(resume_ckpt.get("epoch", 0) or 0)
        if train_steps <= 0:
            train_steps = _parse_train_steps_from_ckpt_path(resume_abs)
        logger.info(
            f"[resume] loaded model/opt  train_steps={train_steps}  start_epoch={start_epoch}  "
            f"ckpt_scale_factor={resume_ckpt.get('scale_factor', 'n/a')}"
        )

    if getattr(args, "ema", False):
        if resume_ckpt is not None and resume_ckpt.get("ema") is not None:
            ema_params = {}
            ema_sd = resume_ckpt["ema"]
            for n, p in model.named_parameters():
                if p.requires_grad:
                    ema_params[n] = ema_sd[n].clone().to(device)
        else:
            ema_params = _init_ema_params(model)
    else:
        ema_params = None

    ema_switch_logged = False
    if ema_params is not None:
        logger.info(
            f"[ema] DiT EMA on: decay={args.ema_decay_before} for train_steps<={args.ema_decay_switch_step}, "
            f"then {args.ema_decay_after}"
        )
    log_steps = 0
    running_loss = 0
    # Latent adversarial: warmup/ramp/t-cutoff stats and one-shot log flags
    adv_warm_logged = False
    adv_ramp_logged = False
    running_d_loss_sum = 0.0
    running_d_loss_count = 0
    running_g_loss_sum = 0.0
    running_g_loss_count = 0
    running_adv_kept_sum = 0
    running_adv_total_sum = 0
    if latent_disc is not None:
        _adv_target_log = float(getattr(args, "latent_adv_coef", 0.0) or 0.0)
        _warm_log = max(0, int(getattr(args, "latent_adv_warmup_steps", 0) or 0))
        _ramp_log = max(0, int(getattr(args, "latent_adv_ramp_steps", 0) or 0))
        _tcut_log = float(getattr(args, "latent_adv_t_cutoff", 1.0) or 1.0)
        _aux_log = float(getattr(args, "latent_adv_aux_coef", 1.0) or 0.0)
        _mode_log = str(getattr(args, "latent_adv_mode", "lddgan") or "lddgan")
        _fake0_log = bool(getattr(args, "latent_adv_aux_fake_zero", False))
        _aux_mode = "real→y" + (" + fake→0" if _fake0_log else " (no fake→0)")
        if _mode_log == "lddgan":
            logger.info(
                f"[latent-adv] schedule(mode=lddgan): warmup={_warm_log} -> ramp={_ramp_log} "
                f"-> coef={_adv_target_log}; aux_label_mse_coef={_aux_log} [{_aux_mode}]; "
                f"t_cutoff={_tcut_log:.3f} "
                f"(discriminate transitions at DDPM t/T < cutoff; label head for partial-label selection)"
            )
        else:
            logger.info(
                f"[latent-adv] schedule(mode=plocgan): warmup={_warm_log} -> ramp={_ramp_log} "
                f"-> coef={_adv_target_log}; aux_label_mse_coef={_aux_log} [{_aux_mode}]; "
                f"t_cutoff={_tcut_log:.3f}"
            )
    # Recon L1: stabilizes x0_hat, complements the GAN
    running_l1_sum = 0.0
    running_l1_count = 0
    running_freq_sum = 0.0
    running_freq_count = 0
    running_lap_sum = 0.0
    running_lap_count = 0
    running_repa_sum = 0.0
    running_repa_count = 0
    running_moment_sum = 0.0
    running_moment_mean_sum = 0.0
    running_moment_std_sum = 0.0
    running_moment_count = 0
    running_ratio_sum = 0.0
    running_ratio_count = 0
    _l1_coef_log = float(getattr(args, "recon_l1_coef", 0.0) or 0.0)
    _moment_coef_log = float(getattr(args, "moment_coef", 0.0) or 0.0)
    _ratio_coef_log = float(getattr(args, "ratio_label_coef", 0.0) or 0.0)
    _moment_std_only_log = bool(getattr(args, "moment_std_only", False))
    _l1_space_log = str(getattr(args, "recon_l1_space", "latent") or "latent").lower()
    _freq_coef_log = float(getattr(args, "recon_freq_coef", 0.0) or 0.0)
    _freq_space_log = str(getattr(args, "recon_freq_space", "image") or "image").lower()
    if _freq_coef_log > 0:
        logger.info(
            f"[recon-freq] coef={_freq_coef_log} space={_freq_space_log} "
            f"ratio={float(getattr(args, 'recon_freq_ratio', 0.3))} "
            f"high_w={float(getattr(args, 'recon_freq_high_weight', 3.0))} "
            f"t_cutoff={float(getattr(args, 'recon_freq_t_cutoff', 0.5))}"
        )
    if _l1_coef_log > 0:
        _diff_type_log = getattr(args, "diffusion_type", "ddpm")
        if _diff_type_log == "flow":
            _l1_weight_note = "weighted by (1-t) (flow x0_hat=x_t-t*v -> 0 at high t)"
        elif diffusion.model_mean_type != ModelMeanType.START_X:
            _l1_weight_note = (
                "weighted by alpha_bar_t (full at low t, ->0 at high t, suppresses eps-solved x0_hat blow-up)"
            )
        else:
            _l1_weight_note = "no t weighting (x0 mode, x0_hat is the model output)"
        logger.info(
            f"[recon-l1] coef={_l1_coef_log} space={_l1_space_log} ("
            f"{'L1 in latent space x0_hat-z_protein' if _l1_space_log == 'latent' else 'L1 in pixel space decode(x0_hat)-protein_img'})"
            f" | {_l1_weight_note}"
        )
    if _moment_coef_log > 0 and getattr(args, "diffusion_type", "ddpm") == "flow":
        _moment_mode = "std only" if _moment_std_only_log else "mean+std"
        logger.info(
            f"[moment] coef={_moment_coef_log} mode={_moment_mode} "
            f"(per-channel spatial L1, flow weight w=1-t; x0_hat=x_t-t*v)"
        )
    if _ratio_coef_log > 0 and getattr(args, "diffusion_type", "ddpm") == "flow":
        logger.info(
            f"[ratio-loss] coef={_ratio_coef_log} "
            f"cell_thr={float(getattr(args, 'ratio_cell_thr', 0.05))} "
            f"nuc_thr={float(getattr(args, 'ratio_nuc_thr', 0.10))} "
            f"w=(1-t)^{float(getattr(args, 'ratio_loss_1mint_weight', 1.0))} "
            f"decode_chunk={int(getattr(args, 'ratio_decode_chunk', 0) or 0)} "
            f"[DAPI-hard-mask mode]"
        )
    # Gradient accumulation (effective batch_size = batch_size * grad_accum_steps)
    _accum = max(1, int(getattr(args, "grad_accum_steps", 1) or 1))
    _accum_pos = 0  # micro-step counter inside the current window (0..accum-1)
    _d_active_window = False  # whether D gradients were accumulated in this window
    if _accum > 1:
        logger.info(
            f"[grad-accum] grad_accum_steps={_accum}, "
            f"effective_batch_size = {args.batch_size}*{_accum} = {args.batch_size * _accum}; "
            f"warmup/ramp/log/ckpt/EMA counters use effective optimizer steps; --log-every={args.log_every} counts those."
        )
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")

    _steps_per_epoch = len(loader) // _accum   # effective optimizer steps per epoch
    if start_epoch > 0:
        _total_steps = train_steps + (args.epochs - start_epoch) * _steps_per_epoch
    else:
        _total_steps = args.epochs * _steps_per_epoch
    _cosine_T_max    = max(1, _total_steps - _lr_warmup_steps)
    if _lr_warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        _warmup_sched = LinearLR(
            opt,
            start_factor=1e-6 / args.lr,   # linear ramp from a tiny LR to peak
            end_factor=1.0,
            total_iters=_lr_warmup_steps,
        )
        _cosine_sched = CosineAnnealingLR(opt, T_max=_cosine_T_max, eta_min=_lr_eta_min)
        _lr_scheduler = SequentialLR(opt, [_warmup_sched, _cosine_sched], milestones=[_lr_warmup_steps])
    else:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        _lr_scheduler = CosineAnnealingLR(opt, T_max=_cosine_T_max, eta_min=_lr_eta_min)
    # On resume, advance the scheduler to the train_steps position
    if train_steps > 0:
        for _ in range(train_steps):
            _lr_scheduler.step()
    logger.info(
        f"[lr-schedule] warmup={_lr_warmup_steps} steps → CosineAnnealingLR("
        f"T_max={_cosine_T_max}, eta_min={_lr_eta_min:.1e}) | "
        f"total={_total_steps} steps ({_steps_per_epoch} steps/epoch × "
        f"{args.epochs - start_epoch} epochs from epoch {start_epoch})"
    )

    # Pre-training single-batch loss probe: flow expects v-MSE ~ (1+var(x0)), eps expects ~1
    with torch.no_grad():
        probe = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True,
                )
            )
        )
        if len(probe) == 6:
            _img0, _label0, _, _, _, _ = probe
        else:
            _img0, _label0, _, _ = probe
        _img0 = _img0.to(device)
        _label0 = _label0.to(device)
        _z = encode_protein_latent(ae, protein_channel(_img0, args.protein_ch_index), scale_factor)
        _s2 = struct_two_channel(_img0, args.struct_ch_indices)
        _zs = struct_encoder(_s2) * struct_scale_factor
        _diff_type_probe = getattr(args, "diffusion_type", "ddpm")
        if _diff_type_probe == "flow":
            _use_repa_p = repa_proj is not None and _repa_coef > 0
            _lv, _hs, _x0_p, _t_f = diffusion_training_loss_flow(
                diffusion, model, _z, dict(z_struct=_zs, y=_label0),
                return_layer=_repa_layer if _use_repa_p else None,
            )
            _probe_loss = _lv.mean().item()
            _loss_note = " (flow: v-MSE, expect ~1+var(z₀) at zero-init)"
            _probe_tail = " (flow-mode)"
            if _use_repa_p:
                from repa import repa_cosine_loss
                _h_proj = repa_proj(_hs)
                with torch.no_grad():
                    _h_teacher = repa_teacher(_repa_teacher_input(_img0, args))
                _probe_loss += _repa_coef * float(
                    repa_cosine_loss(_h_proj, _h_teacher).detach().item()
                )
                _loss_note += f" + repa(coef={_repa_coef})"
            _l1c_p = float(getattr(args, "recon_l1_coef", 0.0) or 0.0)
            if _l1c_p > 0:
                _l1s_p = str(getattr(args, "recon_l1_space", "latent") or "latent").lower()
                if _l1s_p == "latent":
                    _l1_per = mean_flat((_x0_p - _z).abs())
                else:
                    _x_rec_p = decode_protein_latent_train(ae, _x0_p, scale_factor)
                    _l1_per = mean_flat(
                        (_x_rec_p - protein_channel(_img0, args.protein_ch_index)).abs()
                    )
                _loss_l1 = ((1.0 - _t_f).clamp(min=0.0) * _l1_per).mean().item()
                _probe_loss += _l1c_p * _loss_l1
                _loss_note += f" + recon_l1({_l1s_p}, coef={_l1c_p}, flow-w=1-t)"
            _moment_c_p = float(getattr(args, "moment_coef", 0.0) or 0.0)
            if _moment_c_p > 0:
                _loss_m_p, _lm_p, _ls_p = flow_latent_moment_loss(
                    _x0_p, _z, _t_f,
                    std_only=bool(getattr(args, "moment_std_only", False)),
                )
                _probe_loss += _moment_c_p * float(_loss_m_p.detach().item())
                _mm_p = "std-only" if getattr(args, "moment_std_only", False) else "mean+std"
                _loss_note += f" + moment({_mm_p}, coef={_moment_c_p}, w=1-t)"
        else:
            if getattr(args, "stratified_t", True):
                _t = stratified_t_sample(diffusion.num_timesteps, _z.shape[0], device)
            else:
                _t = torch.randint(0, diffusion.num_timesteps, (_z.shape[0],), device=device)
            _noise_p = torch.randn_like(_z)
            _x_t_p = diffusion.q_sample(_z, _t, noise=_noise_p)
            _out_p = model(_x_t_p, _t, **dict(z_struct=_zs, y=_label0))
            _pc_p = _pred_channels_from_model(model)
            if diffusion.model_mean_type == ModelMeanType.START_X:
                _x0_p = _out_p[:, :_pc_p]
                _probe_loss = mean_flat((_z - _x0_p) ** 2).mean().item()
                _loss_note = ""
                _probe_tail = " (x0 mode: MSE(||x0||^2) scales with latent variance; the eps probe does not apply)"
            else:
                _eps_p = _out_p[:, :_pc_p]
                _pe = mean_flat((_noise_p - _eps_p) ** 2)
                _x0_p = diffusion._predict_xstart_from_eps(x_t=_x_t_p, t=_t, eps=_eps_p)
                _gamma = float(getattr(args, "minsnr_gamma", 0.0))
                if _gamma > 0:
                    _probe_loss = (minsnr_weight(diffusion, _t, gamma=_gamma) * _pe).mean().item()
                    _loss_note = f" (Min-SNR γ={_gamma:.1f})"
                else:
                    _probe_loss = _pe.mean().item()
                    _loss_note = ""
                _probe_tail = " (ε-mode: MSE-only expect ~1.0 at DiT zero-init)"
            _l1c_p = float(getattr(args, "recon_l1_coef", 0.0) or 0.0)
            if _l1c_p > 0:
                _l1s_p = str(getattr(args, "recon_l1_space", "latent") or "latent").lower()
                if _l1s_p == "latent":
                    _l1_p = (_x0_p - _z).abs().mean().item()
                else:
                    _x_rec_p = decode_protein_latent_train(ae, _x0_p, scale_factor)
                    _l1_p = (
                        _x_rec_p - protein_channel(_img0, args.protein_ch_index)
                    ).abs().mean().item()
                _probe_loss = _probe_loss + _l1c_p * _l1_p
                _loss_note = (_loss_note or "") + f" + recon_l1({_l1s_p}, coef={_l1c_p})"
    logger.info(
        f"[init-loss] train loss probe (1 batch, no train step)"
        f"{_loss_note}: {_probe_loss:.4f} "
        f"{_probe_tail}; logged train loss is a running avg over log_every steps"
    )

    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")

        for batch in loader:
            # Window start: clear gradients (needed whether or not D participates)
            if _accum_pos == 0:
                opt.zero_grad(set_to_none=True)
                if latent_disc is not None and float(getattr(args, "latent_adv_coef", 0.0) or 0.0) > 0:
                    opt_d.zero_grad(set_to_none=True)
                _d_active_window = False

            # Unpack batch (newer datasets also return is_partial and candidates)
            if len(batch) == 6:
                img, label, confidence, batch_indices, is_partial, candidates = batch
            else:
                # Backward compatibility with older datasets
                img, label, confidence, batch_indices = batch
                is_partial = torch.zeros(img.shape[0], dtype=torch.bool)
                candidates = torch.zeros((0, 2), dtype=torch.float32)

            img = img.to(device)
            label = label.to(device)
            confidence = confidence.to(device)
            batch_indices = batch_indices.to(device)
            is_partial = is_partial.to(device)
            candidates = candidates.to(device)

            protein_img = protein_channel(img, args.protein_ch_index)
            z_protein = encode_protein_latent(ae, protein_img, scale_factor)

            s2 = struct_two_channel(img, args.struct_ch_indices)
            if struct_freeze:
                with torch.no_grad():
                    z_struct = struct_encoder(s2) * struct_scale_factor
            else:
                z_struct = struct_encoder(s2) * struct_scale_factor

            # Partial-label learning: pick candidates with the D label head before CFG dropout.
            # Do not index candidates[0]: non-partial rows are all-zero padding and would set labels to [0,0]
            if (latent_disc is not None
                and is_partial.any()
                and getattr(args, 'enable_partial_label_learning', False)):

                dual_mask = is_partial.bool()
                n_dual = int(dual_mask.sum().item())

                if n_dual > 0:
                    z_prot_dual = z_protein[dual_mask]
                    z_struct_dual = z_struct[dual_mask]
                    cand = torch.as_tensor(
                        PARTIAL_LABEL_CANDIDATES,
                        device=z_prot_dual.device,
                        dtype=z_prot_dual.dtype,
                    )  # (M, 2)

                    with torch.no_grad():
                        if isinstance(latent_disc, LatentLDDGANDiscriminator):
                            tn0 = torch.zeros(
                                n_dual, device=z_prot_dual.device, dtype=z_prot_dual.dtype
                            )
                            _, y_pred_disc = latent_disc(
                                z_prot_dual.detach(),
                                z_prot_dual.detach(),
                                tn0,
                                z_struct_dual.detach(),
                            )
                        else:
                            x_dual = torch.cat(
                                [z_prot_dual.detach(), z_struct_dual.detach()], dim=1
                            )
                            _, y_pred_disc = latent_disc(x_dual)

                        dists = torch.sum(
                            (cand.unsqueeze(0) - y_pred_disc.unsqueeze(1))**2,
                            dim=-1
                        )
                        best_idx = torch.argmin(dists, dim=1)
                        selected_labels = cand[best_idx]
                        label[dual_mask] = selected_labels

                        if train_steps % 500 == 0 and rank == 0:
                            sel_hist = torch.bincount(
                                best_idx.cpu(),
                                minlength=len(cand)
                            ).float() / len(best_idx)
                            logger.info(
                                f"[partial-label] step={train_steps}: "
                                f"selected {n_dual} dual-label samples, "
                                f"distribution={sel_hist.tolist()}"
                            )
                            label_mae = F.l1_loss(y_pred_disc, selected_labels)
                            logger.info(
                                f"[partial-label] D label MAE on dual: {label_mae:.4f}"
                            )

            # Dropout only affects the conditioning; clean labels stay available for
            # the ratio loss and the discriminator label head further below.
            z_struct, label_cond = apply_cfg_train_dropout(
                z_struct, label, float(getattr(args, "cfg_train_dprob", 0.0) or 0.0)
            )

            model_kwargs = dict(z_struct=z_struct, y=label_cond)
            _diff_type = getattr(args, "diffusion_type", "ddpm")
            if _diff_type == "flow":
                _use_repa = (repa_proj is not None and _repa_coef > 0)
                loss_v, h_student, x0_hat, t_flow = diffusion_training_loss_flow(
                    diffusion, model, z_protein, model_kwargs,
                    return_layer=_repa_layer if _use_repa else None,
                )
                # Flow weighting: high-t (noisy) samples have larger v-MSE; decay by (1-t) to avoid domination
                _flow_w_use = float(getattr(args, "flow_loss_1mint_weight", 0.0) or 0.0)
                if _flow_w_use > 0:
                    w_flow = (1.0 - t_flow).clamp(min=_flow_w_use * 0.1)
                    loss = (w_flow * loss_v).mean()
                else:
                    loss = loss_v.mean()
                if _use_repa:
                    from repa import repa_cosine_loss
                    h_proj = repa_proj(h_student)
                    with torch.no_grad():
                        h_teacher = repa_teacher(_repa_teacher_input(img, args))
                    loss_repa = repa_cosine_loss(h_proj, h_teacher)
                    loss = loss + _repa_coef * loss_repa
                    running_repa_sum += float(loss_repa.detach().item())
                    running_repa_count += 1
                _l1_coef = float(getattr(args, "recon_l1_coef", 0.0) or 0.0)
                if _l1_coef > 0:
                    _l1_space = str(getattr(args, "recon_l1_space", "latent") or "latent").lower()
                    if _l1_space == "latent":
                        l1_per = mean_flat((x0_hat - z_protein).abs())
                    elif _l1_space == "image":
                        x_recon = decode_protein_latent_train(ae, x0_hat, scale_factor)
                        l1_per = mean_flat((x_recon - protein_img).abs())
                    else:
                        raise ValueError(f"Unknown --recon-l1-space: {_l1_space}")
                    w_l1 = (1.0 - t_flow).clamp(min=0.0)
                    loss_l1 = (w_l1 * l1_per).mean()
                    loss = loss + _l1_coef * loss_l1
                    running_l1_sum += float(loss_l1.detach().item())
                    running_l1_count += 1
                _moment_coef = float(getattr(args, "moment_coef", 0.0) or 0.0)
                if _moment_coef > 0:
                    loss_moment, _lm, _ls = flow_latent_moment_loss(
                        x0_hat, z_protein, t_flow,
                        std_only=bool(getattr(args, "moment_std_only", False)),
                    )
                    loss = loss + _moment_coef * loss_moment
                    running_moment_sum += float(loss_moment.detach().item())
                    running_moment_mean_sum += _lm
                    running_moment_std_sum += _ls
                    running_moment_count += 1
                _ratio_coef = float(getattr(args, "ratio_label_coef", 0.0) or 0.0)
                if _ratio_coef > 0:
                    mae_ratio = ratio_mae_from_x0_hat_chunked(
                        ae,
                        x0_hat,
                        scale_factor,
                        img,
                        label,
                        args.struct_ch_indices,
                        cell_thr=float(getattr(args, "ratio_cell_thr", 0.05)),
                        nuc_thr=float(getattr(args, "ratio_nuc_thr", 0.10)),
                        decode_chunk=int(getattr(args, "ratio_decode_chunk", 0) or 0),
                    )
                    _rw = float(getattr(args, "ratio_loss_1mint_weight", 1.0) or 0.0)
                    if _rw > 0:
                        w_ratio = torch.pow((1.0 - t_flow).clamp(min=0.0), _rw)
                        loss_ratio = (w_ratio * mae_ratio).mean()
                    else:
                        loss_ratio = mae_ratio.mean()
                    loss = loss + _ratio_coef * loss_ratio
                    running_ratio_sum += float(loss_ratio.detach().item())
                    running_ratio_count += 1
                # flow: t in [0,1], t->0 is closer to clean data
                t_norm = t_flow
            else:
                # DDPM: single DiT forward, optional Min-SNR
                if getattr(args, "stratified_t", True):
                    t = stratified_t_sample(diffusion.num_timesteps, z_protein.shape[0], device)
                else:
                    t = torch.randint(0, diffusion.num_timesteps, (z_protein.shape[0],), device=device)
                noise_b = torch.randn_like(z_protein)
                x_t = diffusion.q_sample(z_protein, t, noise=noise_b)
                _use_repa = (repa_proj is not None and _repa_coef > 0)
                if _use_repa:
                    model_out, h_student = model(
                        x_t, t, return_layer=_repa_layer, **model_kwargs,
                    )
                else:
                    model_out = model(x_t, t, **model_kwargs)
                    h_student = None
                pc = _pred_channels_from_model(model)

                if diffusion.model_mean_type == ModelMeanType.START_X:
                    x0_hat = model_out[:, :pc]
                    loss_vec = mean_flat((z_protein - x0_hat) ** 2)
                else:
                    eps_pred = model_out[:, :pc]
                    loss_vec = mean_flat((noise_b - eps_pred) ** 2)
                    x0_hat = diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps_pred)

                _gamma = float(getattr(args, "minsnr_gamma", 0.0))
                if diffusion.model_mean_type != ModelMeanType.START_X and _gamma > 0:
                    w = minsnr_weight(diffusion, t, gamma=_gamma)
                    loss = (w * loss_vec).mean()
                else:
                    loss = loss_vec.mean()

                # REPA alignment: student hidden layer -> projector -> cosine vs. teacher(GT)
                if _use_repa:
                    from repa import repa_cosine_loss
                    h_proj = repa_proj(h_student)
                    with torch.no_grad():
                        h_teacher = repa_teacher(_repa_teacher_input(img, args))
                    loss_repa = repa_cosine_loss(h_proj, h_teacher)
                    loss = loss + _repa_coef * loss_repa
                    running_repa_sum += float(loss_repa.detach().item())
                    running_repa_count += 1

                # Recon L1 (stabilizes x0_hat, suppresses GAN high-frequency hallucination).
                # In eps mode x0_hat is solved from eps and amplified by 1/sqrt(alpha_bar_t) at high t,
                # so it must be weighted by alpha_bar_t; x0 mode needs no weighting.
                _l1_coef = float(getattr(args, "recon_l1_coef", 0.0) or 0.0)
                if _l1_coef > 0:
                    _l1_space = str(getattr(args, "recon_l1_space", "latent") or "latent").lower()
                    if _l1_space == "latent":
                        l1_per = mean_flat((x0_hat - z_protein).abs())  # (B,)
                    elif _l1_space == "image":
                        x_recon = decode_protein_latent_train(ae, x0_hat, scale_factor)
                        l1_per = mean_flat((x_recon - protein_img).abs())  # (B,)
                    else:
                        raise ValueError(f"Unknown --recon-l1-space: {_l1_space}")
                    if diffusion.model_mean_type != ModelMeanType.START_X:
                        # eps mode: weight by alpha_bar_t; full at low t, decayed at high t
                        ab_t = torch.as_tensor(
                            diffusion.alphas_cumprod, device=t.device, dtype=l1_per.dtype,
                        )[t]
                        loss_l1 = (ab_t * l1_per).mean()
                    else:
                        loss_l1 = l1_per.mean()
                    loss = loss + _l1_coef * loss_l1
                    running_l1_sum += float(loss_l1.detach().item())
                    running_l1_count += 1

                # Frequency reconstruction: FFT magnitude of x0_hat vs. target, weighted by band
                _freq_coef = float(getattr(args, "recon_freq_coef", 0.0) or 0.0)
                if _freq_coef > 0 and _diff_type == "ddpm":
                    _freq_space = str(getattr(args, "recon_freq_space", "image") or "image").lower()
                    _freq_ratio = float(getattr(args, "recon_freq_ratio", 0.3) or 0.3)
                    _freq_hw = float(getattr(args, "recon_freq_high_weight", 3.0) or 3.0)
                    _freq_t_cut = float(getattr(args, "recon_freq_t_cutoff", 0.5) or 0.5)
                    _freq_t_cut = max(0.0, min(1.0, _freq_t_cut))
                    if _freq_space == "latent":
                        pred_f, tgt_f = x0_hat, z_protein
                    elif _freq_space == "image":
                        pred_f = decode_protein_latent_train(ae, x0_hat, scale_factor)
                        tgt_f = protein_img
                    else:
                        raise ValueError(f"Unknown --recon-freq-space: {_freq_space}")
                    freq_per = mean_flat_freq_loss(
                        pred_f, tgt_f,
                        low_freq_ratio=_freq_ratio,
                        low_weight=1.0,
                        high_weight=_freq_hw,
                    )
                    if diffusion.model_mean_type != ModelMeanType.START_X:
                        ab_t = torch.as_tensor(
                            diffusion.alphas_cumprod, device=t.device, dtype=freq_per.dtype,
                        )[t]
                        t_ok = (t.float() / float(diffusion.num_timesteps)) < _freq_t_cut
                        w_freq = ab_t * t_ok.float()
                        denom = w_freq.sum().clamp_min(1e-8)
                        loss_freq = (w_freq * freq_per).sum() / denom
                    else:
                        loss_freq = freq_per.mean()
                    loss = loss + _freq_coef * loss_freq
                    running_freq_sum += float(loss_freq.detach().item())
                    running_freq_count += 1

                # Laplacian edge loss (more local, sensitive to punctate protein texture)
                _lap_coef = float(getattr(args, "recon_lap_coef", 0.0) or 0.0)
                if _lap_coef > 0 and _diff_type == "ddpm":
                    _lap_space = str(getattr(args, "recon_lap_space", "image") or "image").lower()
                    _lap_type = str(getattr(args, "recon_lap_type", "l1") or "l1").lower()
                    _lap_nb = int(getattr(args, "recon_lap_neighborhood", 4) or 4)
                    _lap_t_cut = float(getattr(args, "recon_lap_t_cutoff", 0.5) or 0.5)
                    _lap_t_cut = max(0.0, min(1.0, _lap_t_cut))
                    if _lap_space == "latent":
                        pred_l, tgt_l = x0_hat, z_protein
                    elif _lap_space == "image":
                        pred_l = decode_protein_latent_train(ae, x0_hat, scale_factor)
                        tgt_l = protein_img
                    else:
                        raise ValueError(f"Unknown --recon-lap-space: {_lap_space}")
                    lap_per = mean_flat_laplacian_loss(
                        pred_l, tgt_l, loss_type=_lap_type, neighborhood=_lap_nb,
                    )
                    if diffusion.model_mean_type != ModelMeanType.START_X:
                        ab_t = torch.as_tensor(
                            diffusion.alphas_cumprod, device=t.device, dtype=lap_per.dtype,
                        )[t]
                        t_ok = (t.float() / float(diffusion.num_timesteps)) < _lap_t_cut
                        w_lap = ab_t * t_ok.float()
                        denom_l = w_lap.sum().clamp_min(1e-8)
                        loss_lap = (w_lap * lap_per).sum() / denom_l
                    else:
                        loss_lap = lap_per.mean()
                    loss = loss + _lap_coef * loss_lap
                    running_lap_sum += float(loss_lap.detach().item())
                    running_lap_count += 1
                t_norm = t.float() / float(diffusion.num_timesteps)

            # Latent adversarial: lddgan=transition discrimination, plocgan=endpoint discrimination;
            # warmup -> linear ramp -> full coef, with t-mask keeping low/mid-noise samples only
            _adv_target = float(getattr(args, "latent_adv_coef", 0.0) or 0.0)
            _warm = max(0, int(getattr(args, "latent_adv_warmup_steps", 0) or 0))
            _ramp = max(0, int(getattr(args, "latent_adv_ramp_steps", 0) or 0))
            if _adv_target <= 0 or train_steps < _warm:
                _adv = 0.0
            elif _ramp > 0 and train_steps < _warm + _ramp:
                _adv = _adv_target * float(train_steps - _warm) / float(_ramp)
            else:
                _adv = _adv_target

            if latent_disc is not None and _adv_target > 0:
                if (not adv_warm_logged) and _warm > 0 and train_steps >= _warm:
                    logger.info(
                        f"[latent-adv] step={train_steps}: warmup({_warm}) ended; "
                        f"begin {'ramp over ' + str(_ramp) + ' steps' if _ramp > 0 else 'full coef immediately'}"
                        f" -> coef={_adv_target}"
                    )
                    adv_warm_logged = True
                if (not adv_ramp_logged) and _ramp > 0 and train_steps >= _warm + _ramp:
                    logger.info(
                        f"[latent-adv] step={train_steps}: ramp finished, coef={_adv_target}"
                    )
                    adv_ramp_logged = True

            _t_cut = float(getattr(args, "latent_adv_t_cutoff", 1.0) or 1.0)
            _t_cut = max(0.0, min(1.0, _t_cut))
            _bs_full = int(z_protein.shape[0])
            if latent_disc is not None and _adv > 0 and _t_cut < 1.0:
                keep_mask = t_norm < _t_cut
                n_keep_adv = int(keep_mask.sum().item())
                keep_idx = torch.nonzero(keep_mask, as_tuple=True)[0] if n_keep_adv > 0 else None
            else:
                n_keep_adv = _bs_full if (latent_disc is not None and _adv > 0) else 0
                keep_idx = None

            if latent_disc is not None and _adv > 0:
                running_adv_total_sum += _bs_full
                running_adv_kept_sum += n_keep_adv

            # LDDGAN uses GroupNorm, so N>=1; PLocGAN BN needs N>=2
            _min_keep = 1 if isinstance(latent_disc, LatentLDDGANDiscriminator) else 2
            if latent_disc is not None and _adv > 0 and n_keep_adv >= _min_keep:
                if keep_idx is not None:
                    z_real_k = z_protein[keep_idx]
                    x0_hat_k = x0_hat[keep_idx]
                    zs_k = z_struct[keep_idx]
                    lb_k = label[keep_idx]
                    tn_k = t_norm[keep_idx]
                    if _diff_type == "ddpm":
                        xt_k = x_t[keep_idx]
                        t_k = t[keep_idx]
                else:
                    z_real_k, x0_hat_k, zs_k, lb_k = z_protein, x0_hat, z_struct, label
                    tn_k = t_norm
                    if _diff_type == "ddpm":
                        xt_k, t_k = x_t, t

                requires_grad(model, False)
                requires_grad(latent_disc, True)
                zs_d = zs_k.detach()

                if isinstance(latent_disc, LatentLDDGANDiscriminator):
                    # LDDGAN: true vs. fake posterior + quantitative label head (for partial labels)
                    noise_post = torch.randn_like(z_real_k)
                    x_tm1_real = q_sample_posterior(
                        diffusion, z_real_k.detach(), xt_k.detach(), t_k, noise=noise_post
                    )
                    x_tm1_fake_d = q_sample_posterior(
                        diffusion, x0_hat_k.detach(), xt_k.detach(), t_k, noise=noise_post
                    )
                    logits_r, y_pred_r = latent_disc(x_tm1_real, xt_k.detach(), tn_k, zs_d)
                    logits_f, y_pred_f = latent_disc(x_tm1_fake_d, xt_k.detach(), tn_k, zs_d)
                    loss_d = hinge_discriminator_loss(logits_r, logits_f)
                    _aux_w = float(getattr(args, "latent_adv_aux_coef", 10.0) or 0.0)
                    if _aux_w > 0:
                        _yt = lb_k.float().view(lb_k.shape[0], -1)
                        # Supervise only real transitions -> y by default (stabilizes partial-label selection).
                        # fake->0 pushes the label head to endpoints/zeros, causing MAE~0.25 and candidate collapse.
                        _aux_d = F.mse_loss(y_pred_r, _yt)
                        if bool(getattr(args, "latent_adv_aux_fake_zero", False)):
                            _aux_d = _aux_d + F.mse_loss(y_pred_f, torch.zeros_like(_yt))
                        loss_d = loss_d + _aux_w * _aux_d
                    (loss_d / _accum).backward()
                    _d_active_window = True

                    requires_grad(model, True)
                    requires_grad(latent_disc, False)
                    x_tm1_fake_g = q_sample_posterior(
                        diffusion, x0_hat_k, xt_k.detach(), t_k, noise=noise_post
                    )
                    logits_g, y_pred_g = latent_disc(x_tm1_fake_g, xt_k.detach(), tn_k, zs_d)
                    g_tot = hinge_generator_loss(logits_g)
                    if _aux_w > 0:
                        _yt = lb_k.float().view(lb_k.shape[0], -1)
                        g_tot = g_tot + _aux_w * F.mse_loss(y_pred_g, _yt)
                    loss = loss + _adv * g_tot
                else:
                    # PLocGAN endpoint: D([z_struct, x0_hat]) + optional localization aux
                    x_real_d = torch.cat([z_real_k.detach(), zs_d], dim=1)
                    x_fake_d = torch.cat([x0_hat_k.detach(), zs_d], dim=1)
                    logits_r, y_pred_r = latent_disc(x_real_d)
                    logits_f, y_pred_f = latent_disc(x_fake_d)
                    loss_d = plocgan_discriminator_loss(logits_r, logits_f)
                    _aux_w = float(getattr(args, "latent_adv_aux_coef", 1.0) or 0.0)
                    if _aux_w > 0:
                        _yt = lb_k.float().view(lb_k.shape[0], -1)
                        _aux_d = F.mse_loss(y_pred_r, _yt)
                        if bool(getattr(args, "latent_adv_aux_fake_zero", False)):
                            _aux_d = _aux_d + F.mse_loss(y_pred_f, torch.zeros_like(_yt))
                        loss_d = loss_d + _aux_w * _aux_d
                    (loss_d / _accum).backward()
                    _d_active_window = True

                    requires_grad(model, True)
                    requires_grad(latent_disc, False)
                    x_fake_g = torch.cat([x0_hat_k, zs_d], dim=1)
                    logits_g, y_pred_g = latent_disc(x_fake_g)
                    g_tot = plocgan_generator_loss(logits_g)
                    if _aux_w > 0:
                        _yt = lb_k.float().view(lb_k.shape[0], -1)
                        g_tot = g_tot + _aux_w * F.mse_loss(y_pred_g, _yt)
                    loss = loss + _adv * g_tot

                running_d_loss_sum += float(loss_d.detach().item())
                running_d_loss_count += 1
                running_g_loss_sum += float(g_tot.detach().item())
                running_g_loss_count += 1

                requires_grad(latent_disc, True)

            (loss / _accum).backward()
            running_loss += loss.item() / _accum
            _accum_pos += 1

            # Only at window boundaries: clip + step (model and D), counters, EMA, log, ckpt
            if _accum_pos < _accum:
                # Not at the window end yet: skip step/EMA/log/ckpt (train_steps unchanged)
                continue
            _accum_pos = 0

            _grad_params = list(model.parameters())
            if not struct_freeze:
                _grad_params += list(struct_encoder.parameters())
            torch.nn.utils.clip_grad_norm_(_grad_params, max_norm=10.0)
            opt.step()
            _lr_scheduler.step()

            if _d_active_window:
                torch.nn.utils.clip_grad_norm_(latent_disc.parameters(), max_norm=10.0)
                opt_d.step()
                _d_active_window = False

            log_steps += 1
            train_steps += 1

            if train_steps >= _total_steps:
                if rank == 0:
                    logger.info(
                        f"[train] reached planned total steps {train_steps}/{_total_steps}, stopping."
                    )
                break

            if ema_params is not None:
                _sw = int(args.ema_decay_switch_step)
                _dec = (
                    float(args.ema_decay_before)
                    if train_steps <= _sw
                    else float(args.ema_decay_after)
                )
                if train_steps == _sw + 1 and not ema_switch_logged:
                    logger.info(
                        f"[ema] step={train_steps}: switch decay {args.ema_decay_before} → {args.ema_decay_after} "
                        f"(train_steps > {_sw})"
                    )
                    ema_switch_logged = True
                _ema_update(ema_params, model, _dec)

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                # log_steps counts effective optimizer steps; other running_* counters are per micro-step
                _log_micros = log_steps * _accum
                gan_msg = ""
                if latent_disc is not None and float(getattr(args, "latent_adv_coef", 0.0) or 0.0) > 0:
                    if running_d_loss_count > 0:
                        avg_d = running_d_loss_sum / running_d_loss_count
                        avg_g = running_g_loss_sum / max(running_g_loss_count, 1)
                        kept_pct = (
                            100.0 * running_adv_kept_sum / running_adv_total_sum
                            if running_adv_total_sum > 0 else 0.0
                        )
                        gan_msg = (
                            f" | adv={_adv:.4f} D={avg_d:.3f} G={avg_g:.3f} "
                            f"act={running_d_loss_count}/{_log_micros} "
                            f"kept={kept_pct:.0f}%"
                        )
                    else:
                        gan_msg = f" | adv={_adv:.4f} (warmup/no-active-batch)"
                l1_msg = ""
                if _l1_coef_log > 0 and running_l1_count > 0:
                    avg_l1 = running_l1_sum / running_l1_count
                    l1_msg = f" | l1[{_l1_space_log}]={avg_l1:.4f}"
                freq_msg = ""
                if _freq_coef_log > 0 and running_freq_count > 0:
                    avg_freq = running_freq_sum / running_freq_count
                    freq_msg = f" | freq[{_freq_space_log}]={avg_freq:.4f}"
                lap_msg = ""
                _lap_coef_log = float(getattr(args, "recon_lap_coef", 0.0) or 0.0)
                if _lap_coef_log > 0 and running_lap_count > 0:
                    _lap_space_log = str(getattr(args, "recon_lap_space", "image") or "image").lower()
                    _lap_type_log = str(getattr(args, "recon_lap_type", "l1") or "l1").lower()
                    avg_lap = running_lap_sum / running_lap_count
                    lap_msg = f" | lap[{_lap_space_log},{_lap_type_log}]={avg_lap:.4f}"
                repa_msg = ""
                if _repa_coef > 0 and running_repa_count > 0:
                    avg_repa = running_repa_sum / running_repa_count
                    repa_msg = f" | repa(1-cos)={avg_repa:.4f}"
                moment_msg = ""
                if _moment_coef_log > 0 and running_moment_count > 0:
                    avg_moment = running_moment_sum / running_moment_count
                    avg_m_mean = running_moment_mean_sum / running_moment_count
                    avg_m_std = running_moment_std_sum / running_moment_count
                    moment_msg = f" | moment={avg_moment:.4f}(μ={avg_m_mean:.4f},σ={avg_m_std:.4f})"
                ratio_msg = ""
                if _ratio_coef_log > 0 and running_ratio_count > 0:
                    avg_ratio = running_ratio_sum / running_ratio_count
                    ratio_msg = f" | ratio_mae={avg_ratio:.4f}"
                _cur_lr = _lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.6f} | "
                    f"LR: {_cur_lr:.2e} | Steps/Sec: {steps_per_sec:.2f}"
                    f"{gan_msg}{l1_msg}{freq_msg}{lap_msg}{repa_msg}{moment_msg}{ratio_msg}"
                )
                running_loss = 0
                log_steps = 0
                running_d_loss_sum = 0.0
                running_d_loss_count = 0
                running_g_loss_sum = 0.0
                running_g_loss_count = 0
                running_adv_kept_sum = 0
                running_adv_total_sum = 0
                running_l1_sum = 0.0
                running_l1_count = 0
                running_freq_sum = 0.0
                running_freq_count = 0
                running_lap_sum = 0.0
                running_lap_count = 0
                running_repa_sum = 0.0
                running_repa_count = 0
                running_moment_sum = 0.0
                running_moment_mean_sum = 0.0
                running_moment_std_sum = 0.0
                running_moment_count = 0
                running_ratio_sum = 0.0
                running_ratio_count = 0
                start_time = time.time()

            _ckpt_steps = int(getattr(args, "ckpt_every", 0) or 0)
            if _ckpt_steps > 0 and train_steps % _ckpt_steps == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "train_steps": train_steps,
                    "scale_factor": scale_factor,
                    "struct_scale_factor": struct_scale_factor,
                    "args": args,
                }
                if ema_params is not None:
                    checkpoint["ema"] = _ema_state_dict(model, ema_params)
                if latent_disc is not None:
                    checkpoint["latent_discriminator"] = latent_disc.state_dict()
                    checkpoint["latent_disc_opt"] = opt_d.state_dict()
                if repa_proj is not None:
                    checkpoint["repa_proj"] = repa_proj.state_dict()
                # Frozen pretrained HPA VAE: do not re-save 30MB every time (cnn branch is small)
                if struct_backend != "hpa_vae":
                    checkpoint["struct_encoder"] = struct_encoder.state_dict()
                ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            if train_steps >= _total_steps:
                break

        if train_steps >= _total_steps:
            break

        if rank == 0:
            val_(
                val_loader,
                dataset,
                vis_dataset,
                model,
                ae,
                struct_encoder,
                diffusion,
                diffusion_ddim,
                epoch,
                args,
                device,
                logger,
                vis_dir,
                scale_factor,
                struct_scale_factor,
                ema_params=ema_params,
            )

            _ckpt_epoch = int(getattr(args, "ckpt_every_epoch", 9) or 0)
            _epochs_total = int(args.epochs)
            if _ckpt_epoch > 0 and (
                (epoch + 1) % _ckpt_epoch == 0 or (epoch + 1) == _epochs_total
            ):
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "train_steps": train_steps,
                    "epoch": epoch + 1,
                    "scale_factor": scale_factor,
                    "struct_scale_factor": struct_scale_factor,
                    "args": args,
                }
                if ema_params is not None:
                    checkpoint["ema"] = _ema_state_dict(model, ema_params)
                if latent_disc is not None:
                    checkpoint["latent_discriminator"] = latent_disc.state_dict()
                    checkpoint["latent_disc_opt"] = opt_d.state_dict()
                if repa_proj is not None:
                    checkpoint["repa_proj"] = repa_proj.state_dict()
                if struct_backend != "hpa_vae":
                    checkpoint["struct_encoder"] = struct_encoder.state_dict()
                ckpt_path = f"{checkpoint_dir}/epoch_{epoch + 1:04d}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if rank == 0 and train_steps > 0:
        _ckpt_epoch = int(getattr(args, "ckpt_every_epoch", 9) or 0)
        if _ckpt_epoch > 0:
            _final_ep = min(epoch + 1, int(args.epochs))
            ckpt_path = f"{checkpoint_dir}/epoch_{_final_ep:04d}.pt"
            if not os.path.isfile(ckpt_path):
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "train_steps": train_steps,
                    "epoch": _final_ep,
                    "scale_factor": scale_factor,
                    "struct_scale_factor": struct_scale_factor,
                    "args": args,
                }
                if ema_params is not None:
                    checkpoint["ema"] = _ema_state_dict(model, ema_params)
                if latent_disc is not None:
                    checkpoint["latent_discriminator"] = latent_disc.state_dict()
                    checkpoint["latent_disc_opt"] = opt_d.state_dict()
                if repa_proj is not None:
                    checkpoint["repa_proj"] = repa_proj.state_dict()
                if struct_backend != "hpa_vae":
                    checkpoint["struct_encoder"] = struct_encoder.state_dict()
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved final checkpoint to {ckpt_path}")

    logger.info("Training complete!")
    model.eval()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help=(
            "Dataset root holding train/, val/, trainlabels_{tag}.h5, vallabels_{tag}.h5. "
            "Empty means config/dataset_cypm.yaml or env var PLOCDIT_DATA_ROOT. "
            "Example: --data-root /path/to/CYNP_ALL"
        ),
    )
    parser.add_argument(
        "--dataset-h5-tag",
        type=str,
        default="CyPM",
        help=(
            "Label h5 filename: trainlabels_{tag}.h5 / vallabels_{tag}.h5. "
            "Legacy CYPM uses CyPM; CYNP_ALL uses CyNp; LyMo layouts such as MTLS can leave it empty "
            "to auto-discover LyMo2010Trainlabels_.h5 (RealY)."
        ),
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="auto",
        choices=["auto", "bgy3ch", "bg2ch"],
        help=(
            "Image layout: bgy3ch=BGYImg*.png (nucleus+protein+third marker); "
            "bg2ch=BGImg*.bmp (nucleus+protein only, e.g. MTLS); auto=detect from files under train/."
        ),
    )
    parser.add_argument(
        "--protein-ch-index",
        type=int,
        default=1,
        help="Protein channel index in CHW order (default 1, matching the BG/BGY layout).",
    )
    parser.add_argument(
        "--main-ae-ckpt",
        type=str,
        default="",
        help=(
            f"Main-branch HPA VAE (CELL-Diff) weights; empty uses {DEFAULT_HPA_CKPT} or env var HPA_VAE_PRETRAINED_CKPT"
        ),
    )
    parser.add_argument(
        "--main-ae-backend",
        type=str,
        default="hpa_vae",
        choices=["hpa_vae"],
        help="Legacy compatibility: only hpa_vae is allowed (default). ProLDM/proldm was removed."
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume from this training .pt (restores DiT/opt/EMA/struct_encoder); "
            "the experiment dir is inferred from the path (.../exp_name/checkpoints/xxx.pt) and new ckpts stay there."
        ),
    )
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0],
        help="GPU IDs to use, default: [0] (GPU 0 only)")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=500,
        help="Linear warmup steps (from ~0 up to peak LR); 0 disables warmup; default 500")
    parser.add_argument("--lr-eta-min", type=float, default=1e-5,
        help="CosineAnnealingLR eta_min (final LR of the cosine decay); default 1e-5")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=0,
        help="Checkpoint interval in steps. Default 0 disables it; set explicitly, e.g. --ckpt-every 10000.")
    parser.add_argument("--ckpt-every-epoch", type=int, default=9,
        help="Checkpoint interval in epochs (default 9). The last epoch is always saved; "
             "final weights are also saved when training stops early at the step limit. 0 disables.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Gradient accumulation steps N: effective batch_size = batch_size * N with unchanged VRAM. "
            "warmup/ramp/log/ckpt step counters use effective optimizer steps; EMA also updates only at the boundary."
        ),
    )
    parser.add_argument("--val-every", type=int, default=1,
        help="Run DDIM sampling + visualization every N epochs")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-max-batches", type=int, default=None,
        help="Number of batches averaged for val loss; default None=all (slow); e.g. 50")
    parser.add_argument("--n-samples", type=int, default=4,
        help="Number of samples to visualize per DDIM run")
    parser.add_argument(
        "--vis-split",
        type=str,
        choices=["val", "test"],
        default="test",
        help=(
            "Split used for fixed-index DDIM visualization: test=test/ + testlabels_{tag}.h5 (true hold-out); "
            "val=val/. Falls back to val when test is missing."
        ),
    )
    parser.add_argument(
        "--vis-val-indices",
        type=str,
        default="0,1,2,3",
        help=(
            "Global indices on the --vis-split split, comma separated (default 0,1,2,3). "
            "The name keeps vis-val for compatibility but may index the test set. "
            "Empty disables it, falling back to --vis-train-indices or a random train batch."
        ),
    )
    parser.add_argument(
        "--vis-seq-extra",
        action="store_true",
        default=True,
        help=(
            "After the fixed --vis-val-indices samples, append one more sample per DDIM run "
            "in global test/val order (1st run base only, 2nd run +1, ...)."
        ),
    )
    parser.add_argument(
        "--no-vis-seq-extra",
        action="store_false",
        dest="vis_seq_extra",
        help="Disable sequential appending of hold-out visualization samples.",
    )
    parser.add_argument(
        "--vis-seq-extra-max",
        type=int,
        default=0,
        help="Max columns in the combined image with vis_seq_extra; 0=unlimited (up to the dataset end).",
    )
    parser.add_argument(
        "--vis-train-indices",
        type=str,
        default="",
        help=(
            "Only used when --vis-val-indices is empty: fixed training-set indices for DDIM. "
            "If also empty, a random train batch is drawn each run. Example: --vis-train-indices 0,1,2,3"
        ),
    )
    parser.add_argument(
        "--vis-noise-seed",
        type=int,
        default=DEFAULT_VIS_NOISE_SEED,
        help=(
            "Initial Gaussian noise seed for DDIM visualization (vis only, not training). "
            f"Default {DEFAULT_VIS_NOISE_SEED} keeps it fixed for cross-epoch comparison; "
            "-1 resamples every validation round."
        ),
    )
    parser.add_argument(
        "--train-geom-aug",
        action="store_true",
        help="Training geometry augmentation: joint horizontal/vertical flip and random shift of all channels.",
    )
    parser.add_argument("--aug-hflip-p", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--aug-vflip-p", type=float, default=0.5, help="Vertical flip probability")
    parser.add_argument(
        "--aug-translate-px",
        type=int,
        default=16,
        help="Max random translation in pixels (both directions); 0 disables it",
    )
    parser.add_argument(
        "--noise-schedule",
        type=str,
        default="squaredcos_cap_v2",
        choices=["linear", "squaredcos_cap_v2"],
        help="Noise schedule: linear=linear beta, squaredcos_cap_v2=cosine (Improved DDPM)",
    )
    parser.add_argument("--ddim-steps", type=int, default=250)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument(
        "--ddim-x0-clamp",
        type=float,
        default=4.0,
        help=(
            "Clamp the decoded x0 to [-c, c] during DDIM (c<=0 disables). "
            "In eps mode 1/sqrt(alpha_bar_t) amplifies eps error at high t, which easily diverges early in training; "
            "default 4.0 ~ 3.5 sigma of z_protein (std~1.15), safe and blocks error accumulation."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="CFG strength for validation DDIM/flow visualization; 1.0 disables. >1 needs --cfg-train-dprob>0",
    )
    parser.add_argument(
        "--cfg-train-dprob",
        type=float,
        default=0.15,
        help="Probability of zeroing y and z_struct together during training (learns the unconditional branch)",
    )
    parser.add_argument(
        "--struct-encoder-ckpt",
        type=str,
        default="",
        help="StructLatentEncoder weights; empty means random init (pretrain or load later)",
    )
    parser.add_argument(
        "--struct-ch-indices",
        type=int,
        nargs=2,
        default=[0, 2],
        metavar=("CH0", "CH1"),
        help="Two struct CHW indices (default [0,2]=nucleus+microtubule; auto [0,0] for bg2ch when left at default)",
    )
    parser.add_argument(
        "--struct-tokens-per-ch",
        type=int,
        default=256,
        help="Total struct tokens (single group: nucleus+microtubule channels patchified together, default 256; patch_h x patch_w searched)",
    )
    parser.add_argument(
        "--label-endpoint-interp",
        action="store_true",
        default=False,
        help="AdaLN label embedding: interpolate between endpoint MLP([1,0]/[0,1]) for middle labels; y~0 still uses MLP([0,0])",
    )
    parser.add_argument(
        "--no-label-endpoint-interp",
        action="store_false",
        dest="label_endpoint_interp",
        help="Disable endpoint interpolation; all labels use the plain MLP mapping",
    )
    parser.add_argument(
        "--train-deterministic-only",
        action="store_true",
        default=False,
        help="Train only on deterministic samples [1,0]/[0,1] (stage-1 warmup)",
    )
    parser.add_argument(
        "--train-all-labels",
        action="store_false",
        dest="train_deterministic_only",
        help="Include partial intermediate-label samples (stage-2 pseudo-label training)",
    )
    parser.add_argument(
        "--partial-label-mode",
        type=str,
        default="random",
        choices=["h5", "random", "pseudo"],
        help="Label source for partial samples: h5 / random (default, legacy behavior) / pseudo",
    )
    parser.add_argument(
        "--pseudo-labels-path",
        type=str,
        default="",
        help="Pseudo-label .csv or .h5 path used when partial_label_mode=pseudo",
    )
    parser.add_argument(
        "--struct-encoder-freeze",
        action="store_true",
        help="Freeze the struct encoder (for pretrained weights); default trains it with the DiT (required without a pretrained ckpt)",
    )
    parser.add_argument(
        "--struct-backend",
        type=str,
        default="cnn",
        choices=["cnn", "hpa_vae"],
        help=(
            "Struct encoder backend: cnn=3 conv layers (light, trainable); "
            "hpa_vae=CELL-Diff HPA pretrained VAE (always frozen, 8 output channels, higher quality). "
            "Switching backends changes the struct_embedder input channels (cnn=2 / hpa_vae=8) and is "
            "incompatible with existing ckpts, requiring training from scratch."
        ),
    )
    parser.add_argument(
        "--hpa-vae-ckpt",
        type=str,
        default="",
        help=f"HPA VAE weight path (shared by struct/main branches, default {DEFAULT_HPA_CKPT})",
    )
    parser.add_argument(
        "--hpa-struct-latent-size",
        type=int,
        default=0,
        help=(
            "Spatial size of the HPA struct encoder output. 0=auto-match main_ae latent_size (recommended). "
            "Set explicitly only in rare cases (e.g. downsampling struct to 32)."
        ),
    )
    parser.add_argument(
        "--z-scale-factor",
        type=float,
        default=0.0,
        help="Explicit latent scale factor (>0 skips full-set statistics and cache); 0=auto-estimate or read cache",
    )
    parser.add_argument(
        "--recompute-scale",
        action="store_true",
        help="Ignore caches and re-estimate protein z and struct z_struct std over the full set, then rewrite caches",
    )
    parser.add_argument(
        "--struct-z-scale-factor",
        type=float,
        default=0.0,
        help="Explicit struct latent scale (>0 skips statistics); 0=auto-estimate or read struct_scale_factor_*.txt",
    )
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="Skip the AE recon / scale sanity logs and sample image at startup",
    )
    parser.add_argument(
        "--diffusion-type",
        type=str,
        default="ddpm",
        choices=["ddpm", "flow"],
        help=(
            "ddpm=standard DDPM/DDIM diffusion (predicts eps); "
            "flow=Flow Matching (predicts velocity v=eps-x0, linear interpolation forward, Euler/Heun ODE sampling)"
        ),
    )
    parser.add_argument(
        "--flow-num-steps",
        type=int,
        default=28,
        help="Flow matching sampling steps (default 28); used only by the DDIM sampler object.",
    )
    parser.add_argument(
        "--flow-loss-1mint-weight",
        type=float,
        default=0.0,
        help=(
            "Flow matching loss weighting: when >0 the loss is multiplied by (1-t)^weight "
            "to stop high-t samples from dominating training. Recommended 1.0 (linear decay). 0 disables."
        ),
    )
    parser.add_argument(
        "--flow-sigma-min",
        type=float,
        default=0.0,
        help="Min sigma during flow sampling; 0=integrate to zero noise, >0 (e.g. 1/N)=Flux-style early stop.",
    )
    parser.add_argument(
        "--flow-use-heun",
        action="store_true",
        help="Use Heun 2nd-order sampling for flow matching (more accurate, 2x NFE); default Euler.",
    )
    parser.add_argument(
        "--flow-time-shift-mu",
        type=float,
        default=0.0,
        help="Flow matching time-shift mu (Flux style); 0=no shift, 0.7~1.16 recommended at high resolution.",
    )
    parser.add_argument(
        "--no-stratified-t",
        action="store_true",
        help="Disable stratified-t (fall back to plain uniform t sampling). Default on: each batch covers the full t axis, stabler logs.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Enable EMA shadow weights for the DiT (requires_grad params, in sync with the optimizer); off by default to save VRAM.",
    )
    parser.add_argument(
        "--ema-decay-before",
        type=float,
        default=0.999,
        help="EMA decay while train_steps <= --ema-decay-switch-step (default 0.999).",
    )
    parser.add_argument(
        "--ema-decay-after",
        type=float,
        default=0.9995,
        help="EMA decay after the switch step (default 0.9995).",
    )
    parser.add_argument(
        "--ema-decay-switch-step",
        type=int,
        default=50000,
        help="Use ema-decay-before up to and including this step, ema-decay-after beyond it (default 50000).",
    )
    parser.add_argument(
        "--latent-adv-coef",
        type=float,
        default=0.0,
        help=(
            ">0 enables latent adversarial training. Default mode=lddgan: on DDPM discriminate transitions "
            "D(x_{t-1},x_t,t,z_struct) (true posterior vs. fake posterior from x0_hat). "
            "mode=plocgan discriminates the endpoint D([z_struct,x0_hat]). 0 disables"
        ),
    )
    parser.add_argument(
        "--latent-adv-mode",
        type=str,
        default="lddgan",
        choices=["lddgan", "plocgan"],
        help=(
            "lddgan=intermediate transition discrimination (recommended; forces DDPM; label head for partial labels); "
            "plocgan=endpoint latent discrimination + optional localization head"
        ),
    )
    parser.add_argument(
        "--latent-adv-lr",
        type=float,
        default=2e-4,
        help="AdamW learning rate of the latent discriminator (slightly below the DiT by default)",
    )
    parser.add_argument(
        "--latent-adv-base-ch",
        type=int,
        default=64,
        help="Base channel count of the discriminator first Conv layer",
    )
    parser.add_argument(
        "--latent-adv-label-dim",
        type=int,
        default=32,
        help="(p_cyt,p_nuc) -> broadcast feature dim, concatenated to the latent channels",
    )
    parser.add_argument(
        "--latent-adv-warmup-steps",
        type=int,
        default=0,
        help=(
            "GAN warmup: while train_steps < this value latent_adv_coef is forced to 0 (diffusion loss only), "
            "letting the DiT learn basic structure before the discriminator kicks in. 0=no warmup (legacy)."
        ),
    )
    parser.add_argument(
        "--latent-adv-ramp-steps",
        type=int,
        default=0,
        help=(
            "GAN ramp: steps to linearly raise the effective coef from 0 to latent_adv_coef after warmup. "
            "0=full coef immediately after warmup (not recommended); keep it comparable to warmup."
        ),
    )
    parser.add_argument(
        "--latent-adv-t-cutoff",
        type=float,
        default=1.0,
        help=(
            "GAN t-mask: compute the D/G hinge only for samples with t_norm < cutoff. "
            "flow: t_norm is the continuous t in [0,1]; DDPM: t_norm=t/T. "
            "1.0=adversarial at all timesteps (no cut); <1 only low-noise steps participate, "
            "reducing high-frequency hallucination of x0_hat at high noise."
        ),
    )
    parser.add_argument(
        "--latent-adv-aux-coef",
        type=float,
        default=10.0,
        help=(
            "MSE coefficient of the quantitative label head (lddgan/plocgan). "
            "D supervises real->y by default; G fake->y. Used for partial-label selection; 0 disables."
        ),
    )
    parser.add_argument(
        "--latent-adv-aux-fake-zero",
        action="store_true",
        help=(
            "Also set the D label target of fake samples to all zeros (legacy recipe). "
            "Tends to collapse to endpoints with MAE~0.25; off by default."
        ),
    )
    parser.add_argument(
        "--enable-partial-label-learning",
        action="store_true",
        help=(
            "Enable partial-label learning: pick the best label from the candidate set with the D label head. "
            "Requires --partial-label-mode=random and --latent-adv-aux-coef>0."
        ),
    )
    parser.add_argument(
        "--recon-l1-coef",
        type=float,
        default=0.0,
        help=(
            "L1 reconstruction regularizer on the predicted clean latent x0_hat: loss += coef * ||x0_hat - z_protein||_1 "
            "(latent) or coef * ||AE.decode(x0_hat) - protein_img||_1 (image). "
            "0 disables; in eps mode x0_hat is solved from eps, in x0 mode it is the model output. "
            "Key regularizer for stable GAN training and suppressing high-frequency hallucination."
        ),
    )
    parser.add_argument(
        "--recon-l1-space",
        type=str,
        default="latent",
        choices=["latent", "image"],
        help=(
            "L1 space: latent=L1 on x0_hat - z_protein (no AE.decode, saves VRAM); "
            "image=L1 on AE.decode(x0_hat) - protein_img (more direct but ~2x slower)."
        ),
    )
    parser.add_argument(
        "--moment-coef",
        type=float,
        default=0.0,
        help=(
            "flow only: L1 on the per-channel spatial mean/std of x0_hat=x_t-t*v vs. z_protein, "
            "weighted by w=(1-t); loss += coef * moment_loss. 0 disables; try starting at 0.02."
        ),
    )
    parser.add_argument(
        "--moment-std-only",
        action="store_true",
        help="flow with --moment-coef>0: align std only, not mean (try when fake latents are over-dispersed).",
    )
    parser.add_argument(
        "--ratio-label-coef",
        type=float,
        default=0.0,
        help=(
            "flow only: decode the one-step x0_hat=x_t-t*v and compute nucleus/cytoplasm shares, "
            "then MAE against the input label; loss += coef * ratio_mae. 0 disables."
        ),
    )
    parser.add_argument(
        "--ratio-mask-power",
        type=float,
        default=2.0,
        help="Exclusive soft-mask parameter p of the ratio loss (default 2.0).",
    )
    parser.add_argument(
        "--ratio-mask-temp",
        type=float,
        default=1.5,
        help="Exclusive soft-mask temperature of the ratio loss (default 1.5).",
    )
    parser.add_argument(
        "--ratio-cell-thr",
        type=float,
        default=0.05,
        help="Cell-region threshold of the ratio loss (DAPI+ER sum in [0,1], default 0.05).",
    )
    parser.add_argument(
        "--ratio-nuc-thr",
        type=float,
        default=0.10,
        help="DAPI hard-mask threshold: pixels with DAPI in [0,1] above this are nucleus (default 0.10).",
    )
    parser.add_argument(
        "--ratio-loss-1mint-weight",
        type=float,
        default=1.0,
        help=(
            "Flow time-weight exponent of the ratio loss: >0 weights by (1-t)^w; "
            "1.0=linear decay, 0=no weighting."
        ),
    )
    parser.add_argument(
        "--ratio-decode-chunk",
        type=int,
        default=0,
        help=(
            "Decode chunk size for the ratio loss (AE.decode split along the batch dim) "
            "to lower peak VRAM; 0=whole batch (default). E.g. 2 for batch=8."
        ),
    )
    parser.add_argument(
        "--recon-freq-coef",
        type=float,
        default=0.0,
        help=(
            "Frequency loss on x0_hat vs. target: MSE on the FFT magnitude spectrum with a circular low/high mask, loss += coef * freq_loss. "
            "0 disables; 0.02~0.1 recommended, combinable with --recon-l1-coef."
        ),
    )
    parser.add_argument(
        "--recon-freq-space",
        type=str,
        default="image",
        choices=["latent", "image"],
        help="Space of the frequency loss; image (decoded protein channel) is recommended for fluorescence detail.",
    )
    parser.add_argument(
        "--recon-freq-ratio",
        type=float,
        default=0.3,
        help=(
            "Low-frequency circular mask radius = ratio * min(H,W)/2. "
            ">0 is manual; <=0 auto-estimates from the radial power spectrum of training images (or latents) "
            "before training (see --recon-freq-energy-frac)."
        ),
    )
    parser.add_argument(
        "--recon-freq-energy-frac",
        type=float,
        default=0.85,
        help="Spectral energy fraction captured by the low-frequency circle when auto-estimating ratio (default 0.85).",
    )
    parser.add_argument(
        "--recon-freq-auto-samples",
        type=int,
        default=512,
        help="Number of training images sampled when auto-estimating ratio.",
    )
    parser.add_argument(
        "--recon-freq-high-weight",
        type=float,
        default=3.0,
        help="Weight of the high-frequency term relative to the low-frequency term (default 3.0).",
    )
    parser.add_argument(
        "--recon-freq-t-cutoff",
        type=float,
        default=0.5,
        help=(
            "In eps mode the frequency loss is computed only for t/T < cutoff and weighted by alpha_bar_t; "
            "ignored in x0 mode. 1.0=all t."
        ),
    )
    parser.add_argument(
        "--recon-lap-coef",
        type=float,
        default=0.0,
        help=(
            "Laplacian edge loss coefficient (3x3 high-pass, sensitive to punctate/membrane texture): loss += coef * lap_loss. "
            "0 disables; 0.02~0.1 recommended; combinable with --recon-freq-coef."
        ),
    )
    parser.add_argument(
        "--recon-lap-space",
        type=str,
        default="image",
        choices=["latent", "image"],
        help=(
            "Laplacian space: image=Lap diff on AE.decode(x0_hat) vs. protein image (recommended, perceptual); "
            "latent=Lap diff on x0_hat vs. z_protein (saves AE decode, weaker physically)."
        ),
    )
    parser.add_argument(
        "--recon-lap-type",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="Laplacian residual norm: l1 (default, robust to outliers) / l2.",
    )
    parser.add_argument(
        "--recon-lap-neighborhood",
        type=int,
        default=4,
        choices=[4, 8],
        help="Laplacian neighborhood: 4=cross 3x3; 8=includes diagonals.",
    )
    parser.add_argument(
        "--recon-lap-t-cutoff",
        type=float,
        default=0.5,
        help=(
            "In eps mode the Laplacian loss is computed only for t/T < cutoff and weighted by alpha_bar_t; "
            "ignored in x0 mode. 1.0=all t."
        ),
    )
    parser.add_argument(
        "--dit-parameterization",
        type=str,
        default="eps",
        choices=["eps", "x0"],
        help=(
            "eps=predict diffusion noise eps (default); "
            "x0=predict the clean latent x0 (GaussianDiffusion START_X); "
            "flow mode still predicts velocity v=eps-x0. "
            "Min-SNR weighting is ignored in x0 mode."
        ),
    )
    parser.add_argument(
        "--minsnr-gamma",
        type=float,
        default=5.0,
        help="Min-SNR loss weighting gamma (DDPM eps prediction only); <=0 disables. Ignored in x0 / flow mode.",
    )
    parser.add_argument(
        "--repa-coef",
        type=float,
        default=0.0,
        help=(
            "REPA intermediate-layer alignment: loss += coef * (1 - cos(proj(h_DiT), h_SubCell)). "
            "0 disables; 0.05~0.2 recommended. Enabling loads the frozen SubCell plus a small projector."
        ),
    )
    parser.add_argument(
        "--repa-layer",
        type=int,
        default=4,
        help="DiT layer (0-indexed) whose tokens are aligned; 4 is recommended for 12-layer DiT-B.",
    )
    parser.add_argument(
        "--repa-teacher",
        type=str,
        default="subcell",
        choices=["subcell", "hpa"],
        help="REPA teacher: SubCell only (hpa is an alias).",
    )
    parser.add_argument(
        "--subcell-model-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pretrained", "all_channels_MAE-CellS-ProtS-Pool.pth"),
        help="Path to the SubCell pretrained .pth weights.",
    )
    parser.add_argument(
        "--subcell-model-type",
        type=str,
        default="MAE-CellS-ProtS-Pool",
        help="SubCell model type tag (logging only): MAE-CellS-ProtS-Pool / ViT-ProtS-Pool.",
    )
    parser.add_argument(
        "--subcell-channels",
        type=int,
        default=3,
        choices=[1, 3, 4],
        help=(
            "Logical channels fed to SubCell: "
            "1=protein only; 3=[Protein,ER,DNA] (MT zeroed internally); 4=[Protein,MT=0,ER,DNA]. "
            "Channels come from --protein-ch-index and --struct-ch-indices, not the first N img channels."
        ),
    )
    parser.add_argument(
        "--repa-input-size",
        type=int,
        default=448,
        help="SubCell input side (pretrained at 448; patch16 -> 28x28=784 tokens, bilinearly matched to the DiT).",
    )
    parser.add_argument(
        "--repa-teacher-dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="SubCell inference dtype; bf16/fp16 save VRAM.",
    )
    args = parser.parse_args()
    args.sanity_check = not args.no_sanity_check
    args.stratified_t = not args.no_stratified_t
    args.main_ae_ckpt_resolved = _resolve_hpa_weight_arg(
        getattr(args, "main_ae_ckpt", ""),
        parser,
        "--main-ae-ckpt",
    )
    args.hpa_vae_ckpt = _resolve_hpa_weight_arg(
        getattr(args, "hpa_vae_ckpt", ""),
        parser,
        "--hpa-vae-ckpt",
    )
    args.main_ae_backend = "hpa_vae"
    main(args)
