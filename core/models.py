# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional

from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers.attention import AttentionRope
from timm.layers.pos_embed_sincos import RotaryEmbeddingCat


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_tokenwise(x, shift, scale):
    """Token-wise AdaLN modulation.

    x:     (B, N, H)
    shift: (B, N, H)
    scale: (B, N, H)
    """
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """2D float label ``(p_cytoplasm, p_nucleus)`` -> hidden, concatenated with ``t_emb`` for adaLN.

    Default ``endpoint_interp=False``: sin/cos per scalar -> MLP (original path).
    With ``endpoint_interp=True``:
      - precompute emb_cyto=MLP([1,0]), emb_nucl=MLP([0,1]) (training sees only endpoints);
      - any label y: label_emb = p_cyto·emb_cyto + p_nucl·emb_nucl;
      - y≈0 (CFG unconditional): still uses MLP([0,0]), matching training dropout.
    """

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size: int = 256,
        label_scale: float = 1000.0,
        endpoint_interp: bool = False,
    ):
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError("frequency_embedding_size must be even (split across two label dims).")
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.half_freq = frequency_embedding_size // 2
        self.label_scale = float(label_scale)
        self.endpoint_interp = bool(endpoint_interp)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def _mlp_embed(self, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        a0 = y[:, 0] * self.label_scale
        a1 = y[:, 1] * self.label_scale
        f0 = TimestepEmbedder.timestep_embedding(a0, self.half_freq)
        f1 = TimestepEmbedder.timestep_embedding(a1, self.half_freq)
        feat = torch.cat([f0, f1], dim=-1)
        return self.mlp(feat)

    def _endpoint_embs(self, device, dtype):
        emb_cyto = self._mlp_embed(torch.tensor([[1.0, 0.0]], device=device, dtype=dtype))
        emb_nucl = self._mlp_embed(torch.tensor([[0.0, 1.0]], device=device, dtype=dtype))
        return emb_cyto, emb_nucl

    def forward(self, labels):
        y = labels.float()
        if not self.endpoint_interp:
            return self._mlp_embed(y)

        is_zero = y.abs().sum(dim=-1) < 1e-6
        emb_cyto, emb_nucl = self._endpoint_embs(y.device, y.dtype)
        out = y[:, 0:1] * emb_cyto + y[:, 1:2] * emb_nucl
        if is_zero.any():
            out = out.clone()
            out[is_zero] = self._mlp_embed(y[is_zero])
        return out


def cond_from_time_and_label(t_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
    """Concatenate time and label embeddings on the last dim for adaLN, (N, 2*H)."""
    return torch.cat([t_emb, label_emb], dim=-1)


class LabelContextEmbedder(nn.Module):
    """Two-way labels (p_cyto, p_nucl) -> (B, 2, hidden) cross-attention context tokens.

    Each scalar gets its own sine embedding plus MLP and a separate type_embed that
    separates the cytoplasm and nucleus tokens. The CFG unconditional branch uses an
    externally supplied null_label_ctx and bypasses this module.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256, label_scale: float = 1000.0):
        super().__init__()
        self.label_scale = float(label_scale)
        self.freq_dim = freq_dim
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # type embedding separating the p_cyto and p_nucl tokens
        self.cyto_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.nucl_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cyto_type, std=0.02)
        nn.init.normal_(self.nucl_type, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B, 2) → (B, 2, hidden)。"""
        a0 = labels[:, 0].float() * self.label_scale
        a1 = labels[:, 1].float() * self.label_scale
        f0 = TimestepEmbedder.timestep_embedding(a0, self.freq_dim)   # (B, freq_dim)
        f1 = TimestepEmbedder.timestep_embedding(a1, self.freq_dim)   # (B, freq_dim)
        ctx0 = self.proj(f0).unsqueeze(1) + self.cyto_type            # (B, 1, H)
        ctx1 = self.proj(f1).unsqueeze(1) + self.nucl_type            # (B, 1, H)
        return torch.cat([ctx0, ctx1], dim=1)                         # (B, 2, H)


class LabelCrossAttention(nn.Module):
    """Lightweight cross-attention with image tokens as Q and label context (2 tokens) as KV.

    The output projection is zero-initialised, so the module starts as an additive identity.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_q  = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.q_proj  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        # zero-init: image tokens unchanged at step 0 (like zero-initialised adaLN gates)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_img: torch.Tensor, label_ctx: torch.Tensor) -> torch.Tensor:
        """x_img: (B, N, H)；label_ctx: (B, 2, H) → (B, N, H)。"""
        B, N, H = x_img.shape
        nH, dH = self.num_heads, self.head_dim
        q  = self.q_proj(self.norm_q(x_img)).reshape(B, N, nH, dH).transpose(1, 2)
        kv = self.norm_kv(label_ctx)                                        # (B, 2, H)
        k  = self.k_proj(kv).reshape(B, 2, nH, dH).transpose(1, 2)
        v  = self.v_proj(kv).reshape(B, 2, nH, dH).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)                       # (B, nH, N, dH)
        out = out.transpose(1, 2).reshape(B, N, H)
        return self.out_proj(out)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        block_kwargs.pop("attn_type", None)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionRope(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qkv_fused=True,
            num_prefix_tokens=0,
            attn_drop=0.0,
            proj_drop=0.0,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # conditioning vector c = concat(t_emb, label_emb), dim 2*hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    DiT over channel-concatenated latents (default 6 = 4 noisy protein + 2 struct),
    predicting noise/ε only for the protein subspace (pred_channels=4).

    Conditioning: c = concat(t_emb, label_emb) (2H) drives every block's adaLN.
    Positions: fixed 2D RoPE (timm RotaryEmbeddingCat + AttentionRope), no additive
    sin/cos pos_embed.
    Sampling uses forward (no CFG) by default; forward_with_cfg is an optional
    experiment that doubles the batch.

    Not compatible with old checkpoints using additive pos_embed + timm Attention.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=6,
        pred_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        predict_aux_x0: bool = False,
        cfg_mix_prediction: str = "epsilon",
    ):
        super().__init__()
        if learn_sigma and predict_aux_x0:
            raise ValueError("DiT: learn_sigma and predict_aux_x0 are mutually exclusive (current training uses learn_sigma=False).")
        self.learn_sigma = learn_sigma
        self.predict_aux_x0 = bool(predict_aux_x0)
        self.cfg_mix_prediction = str(cfg_mix_prediction or "epsilon").lower()
        self.in_channels = in_channels
        self.pred_channels = pred_channels
        if learn_sigma:
            self.out_channels = pred_channels * 2
        elif self.predict_aux_x0:
            # first C channels: ε; last C: clean x₀ (same shape as x_start). Sampling uses only ε.
            self.out_channels = pred_channels * 2
        else:
            self.out_channels = pred_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.label_embedder = LabelEmbedder(hidden_size)

        head_dim = hidden_size // num_heads
        gh, gw = self.x_embedder.grid_size
        self.rope_embed = RotaryEmbeddingCat(
            dim=head_dim,
            feat_shape=[gh, gw],
            in_pixels=False,
            temperature=10000.0,
            grid_indexing="ij",
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # LabelEmbedder mirrors TimestepEmbedder.mlp: normal(0.02) on first and last layers
        nn.init.normal_(self.label_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.label_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.label_embedder.mlp[0].bias)
        nn.init.zeros_(self.label_embedder.mlp[2].bias)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, **kwargs):
        """
        x: (N, in_channels, H, W) already concatenated [z_noisy_protein(4), struct_latent(2)]
        t: (N,)
        y: ``(N, 2)`` float; defaults to zeros as a placeholder.
        """
        if y is None:
            y = kwargs.pop('y', None)
        if y is None:
            y = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)

        x_tokens = self.x_embedder(x)
        rope = self.rope_embed.get_embed().to(device=x_tokens.device, dtype=x_tokens.dtype)

        t_emb = self.t_embedder(t)
        label_emb = self.label_embedder(y)
        c = cond_from_time_and_label(t_emb, label_emb)

        for block in self.blocks:
            x_tokens = block(x_tokens, c, rope=rope)
        x_tokens = self.final_layer(x_tokens, c)
        return self.unpatchify(x_tokens)

    def forward_with_cfg(self, x, t, y=None, cfg_scale=None, **kwargs):
        """
        x: (N, in_channels, H, W) with N=2B; the first B rows are the conditional batch,
        the last B have zeroed struct channels (concatenated by the wrapper).
        """
        _y = kwargs.pop('y', None)
        y = y if y is not None else _y
        cfg_scale = cfg_scale if cfg_scale is not None else kwargs.pop('cfg_scale', 1.0)

        half = x.shape[0] // 2

        x_tokens = self.x_embedder(x)
        rope = self.rope_embed.get_embed().to(device=x_tokens.device, dtype=x_tokens.dtype)

        t_emb = self.t_embedder(t)
        if y is not None:
            label_emb = self.label_embedder(y)
        else:
            label_emb = torch.zeros(
                x.shape[0], self.hidden_size, device=t_emb.device, dtype=t_emb.dtype
            )
        c = cond_from_time_and_label(t_emb, label_emb)

        for block in self.blocks:
            x_tokens = block(x_tokens, c, rope=rope)
        x_tokens = self.final_layer(x_tokens, c)
        model_out = self.unpatchify(x_tokens)

        eps_all = model_out[:, : self.pred_channels]
        eps_cond = eps_all[:half]
        eps_uncond = eps_all[half:]

        eps_cond_scaled = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        eps_full = torch.cat([eps_cond_scaled, eps_cond_scaled], dim=0)

        if self.learn_sigma:
            rest = model_out[:half, self.pred_channels :]
            return torch.cat([eps_full, rest], dim=1)
        elif self.predict_aux_x0:
            x0_all = model_out[:, self.pred_channels : self.pred_channels * 2]
            x0_cond = x0_all[:half]
            x0_uncond = x0_all[half:]
            x0_scaled = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            x0_full = torch.cat([x0_scaled, x0_scaled], dim=0)
            return torch.cat([eps_full, x0_full], dim=1)
        elif self.cfg_mix_prediction == "x0":
            x0_all = model_out[:, : self.pred_channels]
            x0_cond = x0_all[:half]
            x0_uncond = x0_all[half:]
            x0_scaled = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            return torch.cat([x0_scaled, x0_scaled], dim=0)
        else:
            return eps_full


class StructPatchEmbedder(nn.Module):
    """Patchify + MLP over struct latents -> hidden_size.

    Default num_groups=1: nucleus + tubulin latents are channel-concatenated and
    patchified once (e.g. HPA ``[8,64,64]`` -> ``M=256`` tokens). num_groups>1 uses
    several independent encoders whose outputs are concatenated.

    Args:
      num_groups          : number of patchify groups (1 recommended)
      latent_ch_per_group : input channels per group (8 when hpa_vae latents merge)
      input_size          : latent spatial size (e.g. 64)
      patch_size          : int or (ph, pw) tuple

    Input: ``(B, num_groups × latent_ch_per_group, H, W)``
    Output: ``(B, num_groups × M, hidden_size)``
    """

    def __init__(
        self,
        num_groups: int,
        latent_ch_per_group: int,
        input_size: int,
        patch_size,                 # int or (ph, pw)
        hidden_size: int,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        if isinstance(patch_size, (list, tuple)):
            self.patch_h, self.patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            self.patch_h = self.patch_w = int(patch_size)
        self.num_groups          = num_groups
        self.latent_ch_per_group = latent_ch_per_group
        self.struct_in_ch        = num_groups
        gh = input_size // self.patch_h
        gw = input_size // self.patch_w
        self.grid_size   = (gh, gw)   # used for RoPE (M positions per group)
        self.num_patches = gh * gw    # M (single group)

        flat_dim  = latent_ch_per_group * self.patch_h * self.patch_w
        inner_dim = hidden_size * mlp_ratio

        # fully independent MLP per group (no weight sharing)
        self.ch_encoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(flat_dim, elementwise_affine=True),
                nn.Linear(flat_dim, inner_dim, bias=True),
                nn.GELU(),
                nn.Linear(inner_dim, hidden_size, bias=True),
            )
            for _ in range(num_groups)
        ])

        self.ch_type_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_size))
            for _ in range(num_groups)
        ])
        for p in self.ch_type_embeds:
            nn.init.normal_(p, std=0.02)

    def _patchify_group(self, z_group: torch.Tensor) -> torch.Tensor:
        """Multi-channel group (B, C, H, W) -> (B, M, C×Ph×Pw), row-major."""
        B, C, H, W = z_group.shape
        Ph, Pw = self.patch_h, self.patch_w
        Gh, Gw = H // Ph, W // Pw
        z = z_group.reshape(B, C, Gh, Ph, Gw, Pw)
        z = z.permute(0, 2, 4, 1, 3, 5).reshape(B, Gh * Gw, C * Ph * Pw)
        return z  # (B, M, C*Ph*Pw)

    def forward(self, z_struct: torch.Tensor) -> torch.Tensor:
        """z_struct: (B, num_groups×C, H, W) -> (B, num_groups×M, hidden_size), seq-concatenated."""
        C = self.latent_ch_per_group
        group_tokens = []
        for i, enc in enumerate(self.ch_encoders):
            z_g     = z_struct[:, i * C : (i + 1) * C]    # (B, C, H, W)
            patches = self._patchify_group(z_g)             # (B, M, C*Ph*Pw)
            tokens  = enc(patches) + self.ch_type_embeds[i] # (B, M, H)
            group_tokens.append(tokens)
        return torch.cat(group_tokens, dim=1)               # (B, num_groups*M, H)


class DiTWithStructSeq(nn.Module):
    """Struct injection via in-context conditioning (sequence concatenation).

      - core.in_channels must equal pred_channels (noise channels only)
      - Struct uses its own PatchEmbed, not the shared noise x_embedder
      - Noise and struct tokens each get fixed 2D RoPE (timm RotaryEmbeddingCat on an
        integer grid), plus a learnable struct type embedding
      - Every DiTBlock self-attn sees both image and struct tokens
      - final_layer applies to image tokens only
      - CFG training replaces struct tokens of unconditional samples with a learnable
        ``null_struct_embed`` instead of zeros, avoiding a mixed "zeros + type_embed"
        signal

    Cost: seq_len grows from N to N+Ns (attention ~4x when Ns=N).
    """

    def __init__(
        self,
        core: DiT,
        struct_in_ch: int = 1,
        struct_latent_ch_per_group: int = 8,
        struct_input_size: int = 32,
        struct_patch_size: int = 2,
        label_endpoint_interp: bool = False,
    ):
        """
        struct_in_ch                : patchify groups (default 1: nucleus+tubulin merged)
        struct_latent_ch_per_group  : latent channels per group (8 for hpa_vae, 2 for cnn)
        """
        super().__init__()
        assert core.in_channels == core.pred_channels, (
            "DiTWithStructSeq requires core.in_channels == pred_channels; "
            "pass in_channels=pred_channels when building DiT (no channel concat)"
        )
        self.core = core
        hidden = core.hidden_size

        self.struct_embedder = StructPatchEmbedder(
            num_groups=struct_in_ch,
            latent_ch_per_group=struct_latent_ch_per_group,
            input_size=struct_input_size,
            patch_size=struct_patch_size,
            hidden_size=hidden,
        )
        head_dim = hidden // core.num_heads
        sgh, sgw = self.struct_embedder.grid_size
        self.struct_rope_embed = RotaryEmbeddingCat(
            dim=head_dim,
            feat_shape=[sgh, sgw],
            in_pixels=False,
            temperature=10000.0,
            grid_indexing="ij",
        )
        # noise token type embedding (struct tokens get theirs in StructPatchEmbedder)
        self.noise_type_embed = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.noise_type_embed, std=0.02)
        self.struct_latent_ch_total = int(struct_in_ch * struct_latent_ch_per_group)

        self.label_embedder = LabelEmbedder(hidden, endpoint_interp=label_endpoint_interp)

        n_blocks = len(core.blocks)
        # struct-token LayerNorms, independent of adaLN (no timestep/label modulation)
        self.struct_norm1 = nn.ModuleList([
            nn.LayerNorm(hidden, elementwise_affine=True, eps=1e-6) for _ in range(n_blocks)
        ])
        self.struct_norm2 = nn.ModuleList([
            nn.LayerNorm(hidden, elementwise_affine=True, eps=1e-6) for _ in range(n_blocks)
        ])
        # mask-aware AdaLN: per-layer extra nucleus/cytoplasm modulation for image tokens
        self.region_adaLN_modulation = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden, 12 * hidden, bias=True),  # 2 regions × 6 params
            )
            for _ in range(n_blocks)
        ])
        # predict the nucleus/cytoplasm soft mask (2 ch) from the struct latent
        self.region_mask_head = nn.Conv2d(self.struct_latent_ch_total, 2, kernel_size=1, bias=True)

        # replace block/final_layer adaLN to accept concat(t_emb, label_emb) (2H)
        for blk in core.blocks:
            blk.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden, 6 * hidden, bias=True),
            )
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        core.final_layer.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden, bias=True),
        )
        nn.init.constant_(core.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(core.final_layer.adaLN_modulation[-1].bias, 0)

        # zero-init struct LayerNorms so cond_struct does not perturb early training
        for ln in self.struct_norm1:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
        for ln in self.struct_norm2:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
        for m in self.region_adaLN_modulation:
            nn.init.constant_(m[-1].weight, 0)
            nn.init.constant_(m[-1].bias, 0)
        nn.init.constant_(self.region_mask_head.weight, 0)
        nn.init.constant_(self.region_mask_head.bias, 0)
        self.use_region_adaln = True

    def _region_masks_from_struct_latent(self, z_struct: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Struct latent -> token-level soft mask (B,N,2) with channels [nucleus, cytoplasm]."""
        b, _, _, _ = z_struct.shape
        logits = self.region_mask_head(z_struct)  # (B,2,H,W)
        probs = F.softmax(logits, dim=1)
        gh, gw = self.core.x_embedder.grid_size
        probs_t = F.adaptive_avg_pool2d(probs, (gh, gw))  # (B,2,gh,gw)
        masks = probs_t.permute(0, 2, 3, 1).reshape(b, gh * gw, 2)
        if masks.shape[1] != n_tokens:
            masks = masks[:, :n_tokens]
        return masks

    def _embed_struct(self, z_struct: torch.Tensor) -> torch.Tensor:
        """Struct latent -> struct tokens.

        Output (B, struct_in_ch × M, H); StructPatchEmbedder already adds each group's
        ch_type_embeds, so nothing extra is marked here.
        """
        return self.struct_embedder(z_struct)   # (B, C×M, H)

    def _forward_seq(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_struct: torch.Tensor,
        y: torch.Tensor,
        return_layer: int | None = None,
    ):
        """Split-stream forward: image tokens use adaLN(cond), struct tokens a separate LN.

        cond = concat(t_emb, label_emb), (B, 2H), driving every block's
        adaLN_modulation(2H->6H). Struct tokens use per-block struct_norm1/2 and are
        not modulated by cond.

        return_layer: None -> only the unpatchified output;
                      int  -> also return the noise-token hidden state (B, N, H) after
                              that block, for REPA-style mid-layer alignment losses.
        """
        x_tokens = self.core.x_embedder(x) + self.noise_type_embed   # (B, N, H)
        s_tokens = self._embed_struct(z_struct)                       # (B, M, H)
        N = x_tokens.shape[1]
        seq = torch.cat([x_tokens, s_tokens], dim=1)                  # (B, N+M, H)

        rope_img        = self.core.rope_embed.get_embed().to(device=seq.device, dtype=seq.dtype)
        rope_struct_one = self.struct_rope_embed.get_embed().to(device=seq.device, dtype=seq.dtype)
        rope_struct = torch.cat(
            [rope_struct_one] * self.struct_embedder.struct_in_ch, dim=0
        )
        rope = torch.cat([rope_img, rope_struct], dim=0)

        if y is None:
            y = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)

        t_emb     = self.core.t_embedder(t)          # (B, H)
        label_emb = self.label_embedder(y)            # (B, H)
        cond        = cond_from_time_and_label(t_emb, label_emb)               # (B, 2H) image tokens
        cond_struct = cond_from_time_and_label(torch.zeros_like(t_emb), label_emb)  # (B, 2H) struct tokens (t=0)
        use_region = bool(getattr(self, "use_region_adaln", True))
        if use_region:
            region_masks = self._region_masks_from_struct_latent(z_struct, N)  # (B, N, 2)
            m_n = region_masks[:, :, 0:1]
            m_c = region_masks[:, :, 1:2]

        captured_hidden = None

        for i, block in enumerate(self.core.blocks):
            x_img = seq[:, :N]   # (B, N, H)
            x_str = seq[:, N:]   # (B, M, H)

            sh_msa,   sc_msa,   g_msa,   sh_mlp,   sc_mlp,   g_mlp   = block.adaLN_modulation(cond).chunk(6, dim=1)
            sh_msa_s, sc_msa_s, g_msa_s, sh_mlp_s, sc_mlp_s, g_mlp_s = block.adaLN_modulation(cond_struct).chunk(6, dim=1)
            if use_region:
                (
                    sh_msa_n, sc_msa_n, g_msa_n, sh_mlp_n, sc_mlp_n, g_mlp_n,
                    sh_msa_c, sc_msa_c, g_msa_c, sh_mlp_c, sc_mlp_c, g_mlp_c,
                ) = self.region_adaLN_modulation[i](cond).chunk(12, dim=1)
                sh_msa_img = sh_msa.unsqueeze(1) + m_n * sh_msa_n.unsqueeze(1) + m_c * sh_msa_c.unsqueeze(1)
                sc_msa_img = sc_msa.unsqueeze(1) + m_n * sc_msa_n.unsqueeze(1) + m_c * sc_msa_c.unsqueeze(1)
                g_msa_img = g_msa.unsqueeze(1) + m_n * g_msa_n.unsqueeze(1) + m_c * g_msa_c.unsqueeze(1)
                sh_mlp_img = sh_mlp.unsqueeze(1) + m_n * sh_mlp_n.unsqueeze(1) + m_c * sh_mlp_c.unsqueeze(1)
                sc_mlp_img = sc_mlp.unsqueeze(1) + m_n * sc_mlp_n.unsqueeze(1) + m_c * sc_mlp_c.unsqueeze(1)
                g_mlp_img = g_mlp.unsqueeze(1) + m_n * g_mlp_n.unsqueeze(1) + m_c * g_mlp_c.unsqueeze(1)
            else:
                sh_msa_img = sh_msa.unsqueeze(1)
                sc_msa_img = sc_msa.unsqueeze(1)
                g_msa_img = g_msa.unsqueeze(1)
                sh_mlp_img = sh_mlp.unsqueeze(1)
                sc_mlp_img = sc_mlp.unsqueeze(1)
                g_mlp_img = g_mlp.unsqueeze(1)

            # self-attention on the joint sequence; img/str use separate norms + adaLN
            x_img_n = modulate_tokenwise(block.norm1(x_img), sh_msa_img, sc_msa_img)
            x_str_n = modulate(self.struct_norm1[i](x_str), sh_msa_s, sc_msa_s)
            attn_out = block.attn(
                torch.cat([x_img_n, x_str_n], dim=1), rope=rope
            )
            x_img = x_img + g_msa_img * attn_out[:, :N]
            x_str = x_str + g_msa_s.unsqueeze(1) * attn_out[:, N:]

            x_img_n = modulate_tokenwise(block.norm2(x_img), sh_mlp_img, sc_mlp_img)
            x_str_n = modulate(self.struct_norm2[i](x_str), sh_mlp_s, sc_mlp_s)
            x_img = x_img + g_mlp_img * block.mlp(x_img_n)
            x_str = x_str + g_mlp_s.unsqueeze(1) * block.mlp(x_str_n)

            seq = torch.cat([x_img, x_str], dim=1)

            if return_layer is not None and i == int(return_layer):
                captured_hidden = x_img        # (B, N, H)

        img_out = self.core.final_layer(seq[:, :N], cond)
        out = self.core.unpatchify(img_out)
        if return_layer is not None:
            return out, captured_hidden
        return out

    def forward(self, x, t, z_struct=None, y=None, return_layer: int | None = None, **kwargs):
        if z_struct is None:
            z_struct = kwargs.pop("z_struct", None)
        if y is None:
            y = kwargs.pop("y", None)
        if z_struct is None:
            raise ValueError("z_struct (struct latent 2ch) is required")
        return self._forward_seq(x, t, z_struct, y, return_layer=return_layer)


class CFGModelWrapper(nn.Module):
    """CFG for DDPM/DDIM sampling: mixes conditional and unconditional predictions.

    Used when the batch is doubled. Flow-matching CFG already happens inside
    ``FlowMatching.ddim_sample_loop``, so this wrapper is not needed there.
    """

    def __init__(self, model: nn.Module, cfg_scale: float = 1.0):
        super().__init__()
        self.model = model
        self.cfg_scale = float(cfg_scale)

    def forward(self, x, t, **kwargs):
        cfg_scale = float(kwargs.pop("cfg_scale", self.cfg_scale))
        half = x.shape[0] // 2
        if cfg_scale <= 1.0 + 1e-6 or half < 1 or x.shape[0] != 2 * half:
            return self.model(x, t, **kwargs)
        out = self.model(x, t, **kwargs)
        if isinstance(out, tuple):
            mixed = self._mix_tensor(out[0], cfg_scale, half)
            return (mixed,) + out[1:]
        return self._mix_tensor(out, cfg_scale, half)

    @staticmethod
    def _mix_tensor(out: torch.Tensor, cfg_scale: float, half: int) -> torch.Tensor:
        c, u = out[:half], out[half:]
        mix = u + cfg_scale * (c - u)
        return torch.cat([mix, mix], dim=0)


def prepare_cfg_ddim_sampling(
    n_samples: int,
    noise: torch.Tensor,
    z_struct: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    diffusion_type: str,
    model: nn.Module,
):
    """Prepare CFG sampling arguments for DDIM / flow ODE.

    Returns:
        shape, noise_in, model_kwargs, sample_model, use_cfg
    """
    cfg_scale = float(cfg_scale)
    use_cfg = cfg_scale > 1.0 + 1e-6
    c, h, w = noise.shape[1:]
    if not use_cfg:
        return (n_samples, c, h, w), noise, dict(z_struct=z_struct, y=y), model, False
    shape = (2 * n_samples, c, h, w)
    noise_in = torch.cat([noise, noise], dim=0)
    mk = dict(z_struct=z_struct, y=y, cfg_scale=cfg_scale)
    if str(diffusion_type).lower() == "flow":
        return shape, noise_in, mk, model, True
    return shape, noise_in, mk, CFGModelWrapper(model, cfg_scale), True


def take_cfg_protein_latent(
    samples: torch.Tensor,
    n_samples: int,
    use_cfg: bool,
    diffusion_type: str,
) -> torch.Tensor:
    """Take the protein latent (B,4,H,W) from the ``ddim_sample_loop`` output."""
    z = samples[:, :4]
    if use_cfg and str(diffusion_type).lower() != "flow":
        z = z[:n_samples]
    return z


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return torch.from_numpy(emb).float()


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
