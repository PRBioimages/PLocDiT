"""Dynamic Frequency-Aware CFG (DFA-CFG) sampling.

Ported from DiT-main's ``FlowMatching`` implementation
(``diffusion/gaussian_diffusion.py``: ``_latent_freq_decompose_*``,
``_facfg_resolve_ratio``, ``_facfg_resolve_weights``,
``_cfg_freq_weighted_mix``, ``_cfg_fresca_mix``), using the same defaults as
``scripts/generate_v2_plocdit_raw_cfglinear.py``: ``fft`` low-pass,
``freq_ratio=0.3``, ``schedule="dual_linear"``.

Rationale: the low-frequency part of a CFG prediction drives layout and
semantics, while the high-frequency part drives texture. DFA-CFG splits the
conditional / unconditional predictions (or their difference) with a circular
FFT low-pass, applies a larger CFG weight on the low band and a smaller one on
the high band, and schedules both the band width and the two weights over
diffusion time ``t`` (``t=1`` = pure noise, ``t=0`` = clean).

Two entry points, with identical math (see :func:`dfa_cfg_mix`):

* flow matching (``FlowMatching``): feed :func:`build_dfa_cfg_kwargs` into
  ``model_kwargs`` and let the sampler perform the mixing internally;
* ddpm / x0 (``SpacedDiffusion``): wrap the denoiser with
  :class:`DFACFGWrapper`, which performs the mixing on the model output.
"""

import math

import torch as th
import torch.nn as nn

# Supported schedules, mirroring DiT-main's ``--cfg-freq-schedule`` values.
VALID_SCHEDULES = (
    "legacy",
    "static",
    "dual_linear",
    "dual_cosine",
    "ratio_anneal",
    "fresca",
    "fresca_ratio_anneal",
)


class DFACFGConfig(object):
    """Hyper-parameters of DFA-CFG.

    Defaults reproduce DiT-main's ``dual_linear`` recipe.

    Parameters
    ----------
    enabled:
        Disable to fall back to plain CFG.
    freq_mode:
        ``"fft"`` (circular FFT low-pass, DiT-main default) or ``"bilinear"``
        (low-resolution down/up-sampling).
    freq_ratio:
        Low-frequency radius, relative to ``min(H, W) / 2``. Larger values
        widen the low band.
    freq_ratio_min:
        Low-band radius at the noise end for ``ratio_anneal`` /
        ``fresca_ratio_anneal``. ``None`` picks ``max(0.05, freq_ratio - 0.15)``.
    freq_tsize:
        Target size of the low-resolution branch in ``"bilinear"`` mode.
    freq_schedule:
        Temporal schedule, one of :data:`VALID_SCHEDULES`.
    time_schedule:
        When ``False`` the weights stay constant over time.
    w_low:
        CFG weight of the low band. ``None`` means "use ``cfg_scale``".
    w_high:
        CFG weight of the high band (``1.0`` leaves high frequencies untouched).
    w_low_min / w_high_min:
        Lower bounds used by the temporal weight schedules.
    """

    def __init__(
        self,
        enabled=True,
        freq_mode="fft",
        freq_ratio=0.3,
        freq_ratio_min=None,
        freq_tsize=(16, 16),
        freq_schedule="dual_linear",
        time_schedule=True,
        w_low=None,
        w_high=1.0,
        w_low_min=1.0,
        w_high_min=1.0,
    ):
        sched = str(freq_schedule or "legacy").lower()
        if sched not in VALID_SCHEDULES:
            raise ValueError(
                "freq_schedule must be one of %s, got %r" % (list(VALID_SCHEDULES), freq_schedule)
            )
        mode = str(freq_mode or "fft").lower()
        if mode not in ("fft", "bilinear"):
            raise ValueError("freq_mode must be 'fft' or 'bilinear', got %r" % (freq_mode,))

        self.enabled = bool(enabled)
        self.freq_mode = mode
        self.freq_ratio = float(freq_ratio)
        self.freq_ratio_min = None if freq_ratio_min is None else float(freq_ratio_min)
        self.freq_tsize = tuple(int(x) for x in (freq_tsize or (16, 16)))
        self.freq_schedule = sched
        self.time_schedule = bool(time_schedule)
        self.w_low = None if w_low is None else float(w_low)
        self.w_high = float(w_high)
        self.w_low_min = float(w_low_min)
        self.w_high_min = float(w_high_min)

    def low_weight(self, cfg_scale):
        """Low-band weight; defaults to ``cfg_scale`` (DiT-main convention)."""
        return float(cfg_scale) if self.w_low is None else float(self.w_low)

    def describe(self, cfg_scale=None):
        """One-line summary for logging."""
        w_low = self.w_low if self.w_low is not None else cfg_scale
        return (
            "DFA-CFG(enabled=%s, mode=%s, schedule=%s, ratio=%.3f, ratio_min=%s, "
            "w_low=%s, w_high=%.3f, w_low_min=%.3f, w_high_min=%.3f, time_schedule=%s)"
            % (
                self.enabled,
                self.freq_mode,
                self.freq_schedule,
                self.freq_ratio,
                "None" if self.freq_ratio_min is None else "%.3f" % self.freq_ratio_min,
                "=cfg_scale" if w_low is None else "%.3f" % float(w_low),
                self.w_high,
                self.w_low_min,
                self.w_high_min,
                self.time_schedule,
            )
        )


# ---------------------------------------------------------------------------
# Frequency decomposition
# ---------------------------------------------------------------------------

def decompose_fft(x, ratio=0.3):
    """Circular FFT low-pass; returns ``(low, high)`` with ``low + high == x``."""
    h, w = x.shape[-2], x.shape[-1]
    ratio = float(max(0.05, min(0.6, ratio)))
    # fftshift places DC at h//2 (not (h-1)/2); keep both masks aligned with it.
    cy, cx = float(h // 2), float(w // 2)
    yy = th.arange(h, device=x.device, dtype=th.float32) - cy
    xx = th.arange(w, device=x.device, dtype=th.float32) - cx
    gy, gx = th.meshgrid(yy, xx, indexing="ij")
    radius = ratio * min(h, w) / 2.0
    mask = ((gy ** 2 + gx ** 2) <= radius ** 2).to(x.dtype).view(1, 1, h, w)

    spec = th.fft.fftshift(th.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
    low = th.fft.ifft2(th.fft.ifftshift(spec * mask, dim=(-2, -1)), dim=(-2, -1))
    low = low.real.to(dtype=x.dtype)
    return low, x - low


def decompose_bilinear(x, tsize=(16, 16)):
    """Bilinear low-pass; returns ``(low, high)`` with ``low + high == x``."""
    h, w = x.shape[-2], x.shape[-1]
    low = th.nn.functional.interpolate(x, size=tuple(tsize), mode="bilinear", align_corners=False)
    low = th.nn.functional.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    return low, x - low


def decompose(x, mode="fft", ratio=0.3, tsize=(16, 16)):
    """Dispatch to the requested frequency decomposition."""
    if str(mode).lower() == "bilinear":
        return decompose_bilinear(x, tsize=tsize)
    return decompose_fft(x, ratio=ratio)


# ---------------------------------------------------------------------------
# Temporal schedules
# ---------------------------------------------------------------------------

def smooth_time(t):
    """Cosine smoothstep mapping of a normalized time."""
    t = float(max(0.0, min(1.0, t)))
    return 0.5 * (1.0 - math.cos(math.pi * t))


def resolve_ratio(freq_ratio, freq_ratio_min, t_cur, freq_schedule):
    """Low-band radius at time ``t_cur``.

    Only the ``*ratio_anneal`` schedules vary the band width: narrow at the
    noise end (``t=1``), wide at the clean end (``t=0``).
    """
    sched = str(freq_schedule or "legacy").lower()
    r_max = float(freq_ratio)
    if sched not in ("ratio_anneal", "fresca_ratio_anneal") or t_cur is None:
        return r_max
    r_min = float(freq_ratio_min) if freq_ratio_min is not None else max(0.05, r_max - 0.15)
    t = float(max(0.0, min(1.0, t_cur)))
    return r_min + (r_max - r_min) * (1.0 - t)


def resolve_weights(w_low, w_high, w_low_min, w_high_min, t_cur, freq_schedule, time_schedule=True):
    """Low / high band CFG weights at time ``t_cur``."""
    w_l, w_h = float(w_low), float(w_high)
    w_l_min, w_h_min = float(w_low_min), float(w_high_min)
    sched = str(freq_schedule or "legacy").lower()

    # Fresca and the ratio-annealing variants always use the cosine schedule.
    if sched in ("fresca", "ratio_anneal", "fresca_ratio_anneal"):
        sched = "dual_cosine"

    if t_cur is None or sched == "static":
        return w_l, w_h
    if not time_schedule:
        return w_l, w_h

    t = float(max(0.0, min(1.0, t_cur)))
    if sched == "dual_cosine":
        t = smooth_time(t)
    if sched in ("dual_linear", "dual_cosine"):
        w_l = w_l_min + (w_l - w_l_min) * t
        w_h = w_h_min + (w_h - w_h_min) * (1.0 - t)
    else:
        # "legacy": only the high band decays, reaching w_high at the noise end.
        w_h = 1.0 + (w_h - 1.0) * t
    return w_l, w_h


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def dfa_cfg_mix(v_c, v_u, cfg_scale, cfg=None, t_cur=None):
    """Mix conditional and unconditional predictions with DFA-CFG.

    Parameters
    ----------
    v_c, v_u:
        Conditional / unconditional predictions with identical shapes. These
        are velocities for flow matching, or eps / x0 for ddpm.
    cfg_scale:
        Plain CFG strength; also the low-band weight when ``cfg.w_low`` is None.
    cfg:
        :class:`DFACFGConfig`. ``None`` or ``enabled=False`` gives plain CFG.
    t_cur:
        Diffusion time normalized to ``[0, 1]`` (``1`` = pure noise).

    Returns
    -------
    Tensor with the same shape as ``v_c``.
    """
    if cfg is None or not cfg.enabled:
        return v_u + float(cfg_scale) * (v_c - v_u)

    w_low = cfg.low_weight(cfg_scale)
    sched = cfg.freq_schedule
    ratio = resolve_ratio(cfg.freq_ratio, cfg.freq_ratio_min, t_cur, sched)

    # Fresca-style: split the CFG delta, then scale each band.
    if sched in ("fresca", "fresca_ratio_anneal"):
        d_lo, d_hi = decompose(v_c - v_u, mode=cfg.freq_mode, ratio=ratio, tsize=cfg.freq_tsize)
        s_l, s_h = resolve_weights(
            w_low, cfg.w_high, cfg.w_low_min, cfg.w_high_min, t_cur, sched, cfg.time_schedule
        )
        return v_u + s_l * d_lo + s_h * d_hi

    # Otherwise split both predictions and run CFG band by band.
    vc_lo, vc_hi = decompose(v_c, mode=cfg.freq_mode, ratio=ratio, tsize=cfg.freq_tsize)
    vu_lo, vu_hi = decompose(v_u, mode=cfg.freq_mode, ratio=ratio, tsize=cfg.freq_tsize)
    w_l, w_h = resolve_weights(
        w_low, cfg.w_high, cfg.w_low_min, cfg.w_high_min, t_cur, sched, cfg.time_schedule
    )
    return (vu_lo + w_l * (vc_lo - vu_lo)) + (vu_hi + w_h * (vc_hi - vu_hi))


def build_dfa_cfg_kwargs(cfg, cfg_scale):
    """Build the ``model_kwargs`` consumed by ``FlowMatching.ddim_sample_loop``.

    The key names are fixed by ``diffusion/gaussian_diffusion.py`` and must not
    be renamed. Returns an empty dict when DFA-CFG is disabled.
    """
    if cfg is None or not cfg.enabled:
        return {}
    kwargs = {
        "cfg_freq_weighted": True,
        "cfg_w_low": float(cfg.low_weight(cfg_scale)),
        "cfg_w_high": float(cfg.w_high),
        "cfg_freq_mode": str(cfg.freq_mode),
        "cfg_freq_ratio": float(cfg.freq_ratio),
        "cfg_freq_tsize": tuple(cfg.freq_tsize),
        "cfg_freq_time_schedule": bool(cfg.time_schedule),
        "cfg_freq_schedule": str(cfg.freq_schedule),
        "cfg_w_low_min": float(cfg.w_low_min),
        "cfg_w_high_min": float(cfg.w_high_min),
    }
    if cfg.freq_ratio_min is not None:
        kwargs["cfg_freq_ratio_min"] = float(cfg.freq_ratio_min)
    return kwargs


def _normalized_time(t, t_max):
    """Scalar representative of a batch of timesteps, normalized to ``[0, 1]``."""
    if t is None:
        return None
    val = float(t.reshape(-1)[0].item()) if th.is_tensor(t) else float(t)
    denom = float(t_max) if t_max and float(t_max) > 0 else 1.0
    return float(max(0.0, min(1.0, val / denom)))


class DFACFGWrapper(nn.Module):
    """Apply DFA-CFG on top of a ddpm / x0 denoiser.

    ``SpacedDiffusion`` doubles the batch to ``2n`` (conditional first,
    unconditional second) but performs no frequency mixing, so the mixing
    happens here, on the model output. The result is returned as
    ``cat([mix, mix])`` to stay compatible with ``take_cfg_protein_latent``.

    Parameters
    ----------
    model:
        The raw DiT denoiser.
    cfg_scale:
        Plain CFG strength (default low-band weight).
    cfg:
        :class:`DFACFGConfig`.
    t_max:
        Timestep normalizer; ``1000`` for ddpm.
    """

    def __init__(self, model, cfg_scale, cfg, t_max=1000.0):
        super(DFACFGWrapper, self).__init__()
        self.model = model
        self.cfg_scale = float(cfg_scale)
        self.cfg = cfg
        self.t_max = float(t_max)

    def forward(self, x, t, **kwargs):
        cfg_scale = float(kwargs.pop("cfg_scale", self.cfg_scale))
        for key in [k for k in kwargs if k.startswith("cfg_")]:
            kwargs.pop(key)

        half = x.shape[0] // 2
        # The DDIM loop already doubles condition tensors; the ancestral
        # p_sample_loop does not, so mirror the DDIM convention here.
        if half >= 1 and x.shape[0] == 2 * half:
            for key, val in list(kwargs.items()):
                if th.is_tensor(val) and val.shape[0] == half:
                    kwargs[key] = th.cat([val, th.zeros_like(val)], dim=0)

        out = self.model(x, t, **kwargs)
        if (
            self.cfg is None
            or not self.cfg.enabled
            or cfg_scale <= 1.0 + 1e-6
            or x.shape[0] != 2 * half
            or half < 1
        ):
            return out

        is_tuple = isinstance(out, tuple)
        pred = out[0] if is_tuple else out
        mix = dfa_cfg_mix(
            pred[:half], pred[half:], cfg_scale, self.cfg, t_cur=_normalized_time(t, self.t_max)
        )
        mix = th.cat([mix, mix], dim=0)
        if not is_tuple:
            return mix
        rest = tuple(
            th.cat([item[:half], item[:half]], dim=0) if th.is_tensor(item) else item
            for item in out[1:]
        )
        return (mix,) + rest
