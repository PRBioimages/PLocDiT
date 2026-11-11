"""SubCell teacher encoder for REPA feature learning.

Frozen SubCell ViT-B/16 producing HPA-style multi-channel patch tokens.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


SUBCELL_FEAT_DIM = 768  # ViT-B hidden size
SUBCELL_INPUT_SIZE = 448
SUBCELL_PATCH_SIZE = 16
SUBCELL_NUM_PATCHES = (SUBCELL_INPUT_SIZE // SUBCELL_PATCH_SIZE) ** 2  # 784


def _remap_subcell_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """SubCell checkpoints use ``encoder.*`` keys; HuggingFace ViTModel has no such prefix."""
    if not isinstance(raw, dict):
        raise TypeError(f"SubCell checkpoint must be a state_dict dict, got {type(raw)}")

    # unwrap nested containers
    for nest in ("state_dict", "model", "encoder"):
        if nest in raw and isinstance(raw[nest], dict):
            nested = raw[nest]
            # descend when the top level holds no ViT tensors
            if nest == "encoder" and any(
                isinstance(v, torch.Tensor) and k.startswith("embeddings")
                for k, v in nested.items()
            ):
                raw = nested
                break
            if nest in ("state_dict", "model"):
                raw = nested
                break

    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        # Lightning weights: encoder.* -> ViTModel; skip auxiliary heads
        if not k.startswith("encoder."):
            continue
        key = k[len("encoder.") :]
        if key.startswith(("pooler.", "classifier.", "decoder.", "head.")):
            continue
        out[key] = v
    return out


class SubCellTeacher(nn.Module):
    """Frozen SubCell ViT-B/16 teacher.

    Accepts 1/3/4 input channels in [-1,1] and reorders them internally to the
    official rybg order [MT, ER, DNA, Protein] (matching subcell-embed COLORS).
    Output patch tokens: (B, 784, 768).
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "MAE-CellS-ProtS-Pool",
        input_size: int = SUBCELL_INPUT_SIZE,
        num_input_channels: int = 3,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.feat_dim = SUBCELL_FEAT_DIM
        self.input_size = int(input_size)
        self.num_input_channels = int(num_input_channels)
        self.model_type = model_type
        self._eval_dtype = dtype

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SubCell weights not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = _remap_subcell_state_dict(checkpoint)

        from transformers import ViTConfig, ViTModel

        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=int(input_size),
            patch_size=SUBCELL_PATCH_SIZE,
            num_channels=4,
        )
        model = ViTModel(config, add_pooling_layer=False)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as _err:
            # Legacy checkpoints may omit unused pooling/head keys; anything else
            # missing means the teacher would stay partially random -> hard error.
            _res = model.load_state_dict(state, strict=False)
            _missing = list(getattr(_res, "missing_keys", []))
            _unexpected = list(getattr(_res, "unexpected_keys", []))
            _allowed = ("head.", "fc.", "classifier.")
            _bad = [k for k in _missing if not k.startswith(_allowed)]
            if _bad:
                raise RuntimeError(
                    f"SubCell teacher weights are incomplete: {_bad[:8]} "
                    f"({len(_missing)} missing, {len(_unexpected)} unexpected). "
                    f"Check that checkpoint keys are encoder.* / ViT format."
                ) from _err
            print(
                f"[SubCellTeacher] loaded with {len(_missing)} head-only keys missing "
                f"(unexpected={len(_unexpected)})"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        self.model = model

        print(
            f"[SubCellTeacher] loaded {model_type} <- {model_path} "
            f"(in_ch={num_input_channels}, feat_dim={self.feat_dim}, "
            f"tokens={self.num_patch_tokens})"
        )

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @property
    def num_patch_tokens(self) -> int:
        side = self.input_size // SUBCELL_PATCH_SIZE
        return side * side

    def _prepare_input(self, x_in: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) in [-1,1] -> (B,4,S,S) in [0,1], channels [MT, ER, DNA, Protein].

        Matches the official subcell-embed ``COLORS=["red","yellow","blue","green"]``.
        """
        if x_in.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x_in.shape)}")
        b, c, h, w = x_in.shape
        x = (x_in.float() + 1.0) * 0.5
        if h != self.input_size or w != self.input_size:
            x = F.interpolate(
                x, size=self.input_size, mode="bilinear", align_corners=False
            )

        if c == 1:
            # protein only: zero MT/ER/DNA
            zeros = torch.zeros(
                b, 3, self.input_size, self.input_size, dtype=x.dtype, device=x.device
            )
            x = torch.cat([zeros, x], dim=1)
        elif c == 3:
            # input is assumed [ER, DNA, Protein] (no MT)
            er, dna, protein = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            mt = torch.zeros(
                b, 1, self.input_size, self.input_size, dtype=x.dtype, device=x.device
            )
            x = torch.cat([mt, er, dna, protein], dim=1)
        elif c == 4:
            # already [MT, ER, DNA, Protein]
            pass
        elif c > 4:
            x = x[:, :4]
        else:
            raise ValueError(f"Unsupported channel count {c}, expected 1/3/4")
        return x

    @torch.no_grad()
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """-> patch tokens (B, N, 768) without CLS."""
        x = self._prepare_input(x_in).to(self._eval_dtype)
        outputs = self.model(pixel_values=x, return_dict=True)
        hidden = outputs.last_hidden_state  # (B, 1+N, 768)
        return hidden[:, 1:, :].float()


def build_subcell_teacher(
    model_path: str,
    model_type: str = "MAE-CellS-ProtS-Pool",
    input_size: int = SUBCELL_INPUT_SIZE,
    num_input_channels: int = 3,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
) -> SubCellTeacher:
    teacher = SubCellTeacher(
        model_path=model_path,
        model_type=model_type,
        input_size=input_size,
        num_input_channels=num_input_channels,
        dtype=dtype,
    )
    return teacher.to(device)


def subcell_feature_dim() -> int:
    return SUBCELL_FEAT_DIM


def repa_teacher_feature_dim(teacher: nn.Module) -> int:
    return int(getattr(teacher, "feat_dim", SUBCELL_FEAT_DIM))
