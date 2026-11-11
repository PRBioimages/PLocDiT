# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Build DiTWithStructSeq (the only supported struct-injection path) from training args."""
from __future__ import annotations

import torch

from models import DiT_models, DiTWithStructSeq
from struct_hpa_encoder import hpa_main_latent_size, hpa_struct_out_channels

IMG_SIZE = 256


def _resolve_main_latent_size(args_ns) -> int:
    return hpa_main_latent_size(IMG_SIZE)


def _resolve_struct_out_ch(args_ns) -> int:
    backend = getattr(args_ns, "struct_backend", "cnn")
    return hpa_struct_out_channels() if backend == "hpa_vae" else 2


def _resolve_struct_spatial(args_ns, main_latent_size: int) -> int:
    backend = getattr(args_ns, "struct_backend", "cnn")
    if backend == "cnn":
        return 32
    _L = getattr(args_ns, "hpa_struct_latent_size", 0) or 0
    return _L if _L > 0 else main_latent_size


def _find_patch_hw(spatial: int, target_tokens: int) -> tuple[int, int]:
    """Find (ph, pw) on a spatial×spatial grid giving exactly target_tokens tokens."""
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
            f"Cannot reach {target_tokens} tokens with integer patches on a {spatial}×{spatial} grid"
        )
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def build_dit_seq_from_args(args_ns, device: torch.device):
    """Return (model, main_L, struct_L).

    Raises for checkpoints recorded with the removed channel_concat mode.
    """
    inject_mode = getattr(args_ns, "struct_inject_mode", None) or "seq_concat"
    if inject_mode == "channel_concat":
        raise ValueError(
            "This checkpoint uses the removed channel_concat struct injection; retrain with seq_concat."
        )
    if inject_mode != "seq_concat":
        raise ValueError(f"Unknown struct_inject_mode: {inject_mode} (only seq_concat)")

    backend = getattr(args_ns, "struct_backend", "cnn")
    struct_out_ch = _resolve_struct_out_ch(args_ns)
    # Nucleus + tubulin latents are already channel-concatenated; one patchify group
    struct_num_groups = 1
    struct_latent_ch_per_group = struct_out_ch

    main_L = _resolve_main_latent_size(args_ns)
    struct_L = _resolve_struct_spatial(args_ns, main_L)

    tokens_per_ch = int(getattr(args_ns, "struct_tokens_per_ch", 0) or 0)
    if tokens_per_ch <= 0:
        tokens_per_ch = 256

    ph, pw = _find_patch_hw(struct_L, tokens_per_ch)
    struct_patch_size = (ph, pw) if ph != pw else ph

    _aux = bool(getattr(args_ns, "dit_predict_aux_x0", False)) and getattr(
        args_ns, "dit_parameterization", "eps"
    ) != "x0"
    _mix = (
        "x0"
        if getattr(args_ns, "dit_parameterization", "eps") == "x0"
        else "epsilon"
    )
    core = DiT_models[args_ns.model](
        input_size=main_L,
        learn_sigma=False,
        in_channels=4,
        pred_channels=4,
        predict_aux_x0=_aux,
        cfg_mix_prediction=_mix,
    )
    model = DiTWithStructSeq(
        core,
        struct_in_ch=struct_num_groups,
        struct_latent_ch_per_group=struct_latent_ch_per_group,
        struct_input_size=struct_L,
        struct_patch_size=struct_patch_size,
        label_endpoint_interp=bool(getattr(args_ns, "label_endpoint_interp", False)),
    ).to(device)
    return model, main_L, struct_L
