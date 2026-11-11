"""Single-image DiT/Flow protein sampling for downstream augmentation (e.g. CGANCode)."""
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

import cv2
import numpy as np
import torch

IMG_SIZE = 256

_DIT_MAIN_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DIT_COV_ROOT = os.path.normpath(os.path.join(_DIT_MAIN_ROOT, "..", "DIT-COV"))

_DIT_MODULE_NAMES = (
    "diffusion",
    "dit_model_build",
    "models",
    "struct_encoder",
    "struct_hpa_encoder",
    "spatial_allocation",
)


def resolve_dit_code_root(ckpt_path: str, code_root: str | None = None) -> str:
    if code_root:
        return os.path.normpath(os.path.abspath(code_root))
    ckpt_path = os.path.abspath(ckpt_path)
    if "DIT-COV" in ckpt_path or "dit-cov" in ckpt_path.lower():
        return _DIT_COV_ROOT
    return _DIT_MAIN_ROOT


def _purge_dit_modules() -> None:
    for name in list(sys.modules):
        if name in _DIT_MODULE_NAMES or any(
            name.startswith(f"{mod}.") for mod in _DIT_MODULE_NAMES
        ):
            del sys.modules[name]


def _import_dit_stack(code_root: str) -> dict[str, ModuleType]:
    code_root = os.path.normpath(os.path.abspath(code_root))
    _purge_dit_modules()
    if code_root in sys.path:
        sys.path.remove(code_root)
    sys.path.insert(0, code_root)
    return {name: importlib.import_module(name) for name in _DIT_MODULE_NAMES if name != "spatial_allocation"}


def _decode_protein(ae, z_scaled: torch.Tensor, scale_factor: float) -> torch.Tensor:
    z = z_scaled / scale_factor
    decoded = ae.decode(z)
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    return decoded


class DitSampler:
    """Load one DiT ckpt and sample protein channel conditioned on struct + 2D label."""

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device | str = "cuda",
        cfg_scale: float = 1.1,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        ddim_x0_clamp: float | None = None,
        seed: int = 0,
        code_root: str | None = None,
    ):
        self.ckpt_path = ckpt_path
        self.code_root = resolve_dit_code_root(ckpt_path, code_root)
        self.device = torch.device(device)
        self.cfg_scale = float(cfg_scale)
        self.ddim_steps = int(ddim_steps)
        self.ddim_eta = float(ddim_eta)
        self.seed = int(seed)

        mods = _import_dit_stack(self.code_root)
        create_diffusion = mods["diffusion"].create_diffusion
        build_dit_seq_from_args = mods["dit_model_build"].build_dit_seq_from_args
        prepare_cfg_ddim_sampling = mods["models"].prepare_cfg_ddim_sampling
        take_cfg_protein_latent = mods["models"].take_cfg_protein_latent
        load_struct_encoder = mods["struct_encoder"].load_struct_encoder
        load_hpa_main_ae = mods["struct_hpa_encoder"].load_hpa_main_ae
        load_hpa_struct_encoder = mods["struct_hpa_encoder"].load_hpa_struct_encoder
        DEFAULT_HPA_CKPT = mods["struct_hpa_encoder"].DEFAULT_HPA_CKPT

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.train_args = ckpt["args"]
        self.scale_factor = float(ckpt.get("scale_factor", 1.0))
        self.struct_scale_factor = float(ckpt.get("struct_scale_factor", 1.0))

        model, main_L, struct_L = build_dit_seq_from_args(self.train_args, self.device)
        if ckpt.get("ema") is not None:
            model.load_state_dict(ckpt["ema"])
        else:
            model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.main_L = main_L

        struct_backend = getattr(self.train_args, "struct_backend", "cnn")
        if struct_backend == "hpa_vae":
            hpa_ckpt = getattr(self.train_args, "hpa_vae_ckpt", "") or DEFAULT_HPA_CKPT
            struct_encoder = load_hpa_struct_encoder(hpa_ckpt, self.device, latent_size=struct_L)
        else:
            struct_ckpt_path = getattr(self.train_args, "struct_encoder_ckpt", "") or None
            struct_encoder = load_struct_encoder(struct_ckpt_path, self.device, torch.float32, freeze=True)
            if "struct_encoder" in ckpt:
                struct_encoder.load_state_dict(ckpt["struct_encoder"])
        struct_encoder.eval()
        for p in struct_encoder.parameters():
            p.requires_grad = False
        self.struct_encoder = struct_encoder

        hpa_main_ckpt = getattr(self.train_args, "hpa_vae_ckpt", "") or DEFAULT_HPA_CKPT
        self.ae = load_hpa_main_ae(hpa_main_ckpt, self.device)

        self.diff_type = getattr(self.train_args, "diffusion_type", "ddpm")
        if self.diff_type == "flow":
            self.diffusion = create_diffusion(
                timestep_respacing=str(self.ddim_steps),
                learn_sigma=False,
                diffusion_type="flow",
                flow_num_steps=self.ddim_steps,
                flow_sigma_min=float(getattr(self.train_args, "flow_sigma_min", 0.0)),
                flow_use_heun=bool(getattr(self.train_args, "flow_use_heun", False)),
                flow_time_shift_mu=float(getattr(self.train_args, "flow_time_shift_mu", 0.0)),
            )
        else:
            predict_xstart = bool(getattr(self.train_args, "predict_xstart", False))
            self.diffusion = create_diffusion(
                timestep_respacing=str(self.ddim_steps),
                learn_sigma=False,
                predict_xstart=predict_xstart,
            )

        if ddim_x0_clamp is not None:
            self.x0_clamp = float(ddim_x0_clamp)
        else:
            self.x0_clamp = float(getattr(self.train_args, "ddim_x0_clamp", 4.0) or 0.0)

        self.struct_ch_indices = tuple(int(x) for x in self.train_args.struct_ch_indices)

        def _denoised_fn(x):
            if self.x0_clamp > 0:
                return x.clamp(-self.x0_clamp, self.x0_clamp)
            return x

        self._denoised_fn = _denoised_fn
        self._prepare_cfg_ddim_sampling = prepare_cfg_ddim_sampling
        self._take_cfg_protein_latent = take_cfg_protein_latent
        self._use_fp16 = False

    def set_fp16(self, enabled: bool = True):
        self._use_fp16 = bool(enabled)
        return self

    def _img_to_tensor(self, img_hwc_01: np.ndarray) -> torch.Tensor:
        if img_hwc_01.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img_u8 = (np.clip(img_hwc_01, 0, 1) * 255).astype(np.uint8)
            img_u8 = cv2.resize(img_u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img_hwc_01 = img_u8.astype(np.float32) / 255.0
        bgr = (np.clip(img_hwc_01, 0, 1) * 255.0).astype(np.uint8)
        t = bgr.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(t.transpose(2, 0, 1))

    @torch.no_grad()
    def sample_protein_batch(
        self,
        imgs_hwc_01: list[np.ndarray],
        label_2: np.ndarray,
        seeds: list[int] | None = None,
    ) -> list[np.ndarray]:
        """Batch sample protein channels; all images share the same 2D label ratio."""
        if not imgs_hwc_01:
            return []
        if seeds is None:
            seeds = [self.seed + i for i in range(len(imgs_hwc_01))]

        batch = torch.stack([self._img_to_tensor(img) for img in imgs_hwc_01], dim=0).to(
            self.device, dtype=torch.float32
        )
        label = np.asarray(label_2, dtype=np.float32).reshape(1, 2)
        y = torch.from_numpy(label).to(self.device, dtype=torch.float32).expand(len(imgs_hwc_01), -1)

        c0, c1 = self.struct_ch_indices
        struct_2ch = batch[:, [c0, c1]].contiguous()
        n = len(imgs_hwc_01)
        noise = torch.empty(n, 4, self.main_L, self.main_L, device=self.device, dtype=torch.float32)
        for i, seed in enumerate(seeds):
            g = torch.Generator(device=self.device).manual_seed(int(seed))
            noise[i] = torch.randn(
                4, self.main_L, self.main_L, device=self.device, dtype=torch.float32, generator=g
            )

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_fp16 and self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            z_struct = self.struct_encoder(struct_2ch) * self.struct_scale_factor

            sample_shape, noise_in, mk, loop_model, use_cfg = self._prepare_cfg_ddim_sampling(
                n, noise, z_struct, y, self.cfg_scale, self.diff_type, self.model
            )
            samples = self.diffusion.ddim_sample_loop(
                loop_model,
                shape=sample_shape,
                noise=noise_in,
                clip_denoised=False,
                model_kwargs=mk,
                denoised_fn=self._denoised_fn,
                progress=False,
                device=self.device,
                eta=self.ddim_eta,
            )
            z_out = self._take_cfg_protein_latent(samples, n, use_cfg, self.diff_type)
            if z_out.shape[0] > n:
                z_out = z_out[:n]
            protein = _decode_protein(self.ae, z_out[:, :4], self.scale_factor)
            protein_01 = ((protein + 1.0) / 2.0).clamp(0.0, 1.0).float()

        out_list = []
        for i in range(n):
            arr = protein_01[i].detach().cpu().numpy()
            # AE decode is (1,H,W) or (H,W); always return HxW for BGY green replace
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise RuntimeError(f"expected HxW protein, got shape={arr.shape}")
            out_list.append(arr)
        return out_list

    @torch.no_grad()
    def sample_protein_batch_sdedit(
        self,
        imgs_hwc_01: list[np.ndarray],
        label_2: np.ndarray,
        strength: float = 0.6,
        seeds: list[int] | None = None,
        protein_ch: int = 1,
    ) -> list[np.ndarray]:
        """GeNIe / SDEdit: partially noise real protein latent, denoise toward label_2.

        strength ∈ (0,1]: fraction of flow time (t≈1 is pure noise). Typical 0.5–0.7
        keeps source low-level protein/structure while shifting semantics via label.
        Only supported for flow diffusion (CyPM ckpt).
        """
        if not imgs_hwc_01:
            return []
        if self.diff_type != "flow":
            raise NotImplementedError("SDEdit path currently requires flow diffusion")
        s = float(strength)
        if not (0.0 < s <= 1.0):
            raise ValueError(f"strength must be in (0,1], got {s}")
        if seeds is None:
            seeds = [self.seed + i for i in range(len(imgs_hwc_01))]

        batch = torch.stack([self._img_to_tensor(img) for img in imgs_hwc_01], dim=0).to(
            self.device, dtype=torch.float32
        )
        label = np.asarray(label_2, dtype=np.float32).reshape(1, 2)
        y = torch.from_numpy(label).to(self.device, dtype=torch.float32).expand(len(imgs_hwc_01), -1)

        c0, c1 = self.struct_ch_indices
        struct_2ch = batch[:, [c0, c1]].contiguous()
        protein_1ch = batch[:, protein_ch : protein_ch + 1].contiguous()
        n = len(imgs_hwc_01)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_fp16 and self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            z_struct = self.struct_encoder(struct_2ch) * self.struct_scale_factor
            enc = self.ae.encode(protein_1ch)
            if hasattr(enc, "sample"):
                z0 = enc.sample()
            elif hasattr(enc, "latent_dist"):
                z0 = enc.latent_dist.mode()
            else:
                z0 = enc
            z0 = z0 * self.scale_factor

            eps = torch.empty_like(z0)
            for i, seed in enumerate(seeds):
                g = torch.Generator(device=self.device).manual_seed(int(seed))
                eps[i] = torch.randn(
                    z0.shape[1:], device=self.device, dtype=z0.dtype, generator=g
                )
            t = torch.full((n,), s, device=self.device, dtype=z0.dtype)
            x_t = self.diffusion.q_sample(z0, t, noise=eps)

            sample_shape, noise_in, mk, loop_model, use_cfg = self._prepare_cfg_ddim_sampling(
                n, x_t, z_struct, y, self.cfg_scale, self.diff_type, self.model
            )
            samples = self.diffusion.ddim_sample_loop(
                loop_model,
                shape=sample_shape,
                noise=noise_in,
                clip_denoised=False,
                model_kwargs=mk,
                denoised_fn=self._denoised_fn,
                progress=False,
                device=self.device,
                eta=self.ddim_eta,
                start_t=s,
            )
            z_out = self._take_cfg_protein_latent(samples, n, use_cfg, self.diff_type)
            if z_out.shape[0] > n:
                z_out = z_out[:n]
            protein = _decode_protein(self.ae, z_out[:, :4], self.scale_factor)
            protein_01 = ((protein + 1.0) / 2.0).clamp(0.0, 1.0).float()

        out = []
        for i in range(n):
            arr = np.squeeze(protein_01[i].detach().cpu().numpy())
            if arr.ndim != 2:
                raise RuntimeError(f"SDEdit expected HxW protein, got {arr.shape}")
            out.append(arr)
        return out

    @torch.no_grad()
    def sample_protein_01(self, img_hwc_01: np.ndarray, label_2: np.ndarray) -> np.ndarray:
        """img: float32 HWC [0,1] BGR; label_2: (1,2) or (2,) ratios. Returns (H,W) in [0,1]."""
        return self.sample_protein_batch([img_hwc_01], label_2, seeds=[self.seed])[0]

    def to(self, device: torch.device | str):
        device = torch.device(device)
        self.device = device
        self.model.to(device)
        self.struct_encoder.to(device)
        self.ae.to(device)
        return self

    def cpu(self):
        return self.to("cpu")
