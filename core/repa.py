"""REPA alignment utilities: SubCell teacher, projector and cosine loss."""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from subcell_teacher import SubCellTeacher, SUBCELL_FEAT_DIM, SUBCELL_INPUT_SIZE

DEFAULT_SUBCELL_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained",
    "all_channels_MAE-CellS-ProtS-Pool.pth",
)


def repa_teacher_feature_dim(teacher: nn.Module) -> int:
    return int(getattr(teacher, "feat_dim", SUBCELL_FEAT_DIM))


def build_repa_teacher(args, device: torch.device) -> nn.Module:
    """Build the frozen SubCell REPA teacher."""
    teacher_type = str(getattr(args, "repa_teacher", "subcell")).strip().lower()
    if teacher_type not in ("subcell", "hpa", ""):
        raise ValueError(
            f"Only the SubCell teacher is supported, got --repa-teacher={teacher_type!r}"
        )

    _dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    _dtype = _dtype_map[str(getattr(args, "repa_teacher_dtype", "fp32")).lower()]

    model_path = str(
        getattr(args, "subcell_model_path", "") or DEFAULT_SUBCELL_CKPT
    ).strip()
    model_type = str(getattr(args, "subcell_model_type", "MAE-CellS-ProtS-Pool"))
    input_size = int(getattr(args, "repa_input_size", SUBCELL_INPUT_SIZE) or SUBCELL_INPUT_SIZE)
    num_channels = int(getattr(args, "subcell_channels", 3))

    teacher = SubCellTeacher(
        model_path=model_path,
        model_type=model_type,
        input_size=input_size,
        num_input_channels=num_channels,
        dtype=_dtype,
    )
    print(
        f"[build_repa_teacher] SubCell: type={model_type}, "
        f"channels={num_channels}, input={input_size}x{input_size}, "
        f"dtype={getattr(args, 'repa_teacher_dtype', 'fp32')}"
    )
    return teacher.to(device)


def align_repa_tokens(tokens: torch.Tensor, target_n: int) -> torch.Tensor:
    """Resize teacher patch tokens to the student token count.

    SubCell (448, patch16) yields 28×28=784; DiT-B/4 with latent 64 yields 16×16=256.
    """
    if tokens.dim() != 3:
        raise ValueError(f"Expected (B,N,C), got {tuple(tokens.shape)}")
    b, n, c = tokens.shape
    if n == target_n:
        return tokens
    side = int(math.sqrt(n))
    target_side = int(math.sqrt(target_n))
    if side * side != n or target_side * target_side != target_n:
        raise ValueError(
            f"Token counts must be perfect squares: teacher N={n}, student N={target_n}"
        )
    x = tokens.transpose(1, 2).reshape(b, c, side, side)
    x = F.interpolate(
        x, size=(target_side, target_side), mode="bilinear", align_corners=False, antialias=True
    )
    return x.flatten(2).transpose(1, 2)


class REPAProjector(nn.Module):
    """REPA projector: 3-layer MLP with zero-initialised output layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = int(hidden_dim or in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def repa_cosine_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """1 - mean cosine similarity for (B, N, C) inputs; auto-aligns differing token counts."""
    if student.shape[-1] != teacher.shape[-1]:
        raise ValueError(
            f"REPA channel mismatch: student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    if student.shape[1] != teacher.shape[1]:
        teacher = align_repa_tokens(teacher, int(student.shape[1]))
    if student.shape != teacher.shape:
        raise ValueError(
            f"REPA shape mismatch: student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    cos = F.cosine_similarity(student.float(), teacher.float(), dim=-1, eps=eps)
    return 1.0 - cos.mean()
