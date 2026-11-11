# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Pixel-domain and latent-space adversarial discriminators.

Training defaults to LDD-GAN style intermediate-state discrimination: real/fake
x₀ go through the same forward noising to form x_t before entering D here; the
endpoint mode is kept as a fallback.

- ``ImagePatchDiscriminator``: single protein channel (B,1,H,W), PatchGAN + hinge.
- ``ImagePlocGANDiscriminator``: PLocGAN-style [struct0, protein, struct1] (B,3,H,W),
  scalar real/fake BCE plus a sigmoid localization-label head.
- Intermediate-state / noise D can append one noise-strength channel.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patchgan_backbone(
    in_ch: int,
    spatial_size: int,
    base_ch: int,
) -> Tuple[nn.Sequential, int]:
    """Shared PatchGAN downsample backbone; returns (backbone, c_feat)."""
    hid = int(base_ch)
    layers: List[nn.Module] = []
    c_cur = int(in_ch)
    sz = int(spatial_size)
    idx = 0
    while sz > 4:
        c_out = min(hid * (2 ** min(idx, 3)), 512)
        layers.append(nn.Conv2d(c_cur, c_out, kernel_size=4, stride=2, padding=1))
        g = max(1, min(8, c_out))
        if c_out % g != 0:
            g = 1
        layers.append(nn.GroupNorm(g, c_out))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        c_cur = c_out
        sz = sz // 2
        idx += 1
        if idx > 16:
            break
    return nn.Sequential(*layers), c_cur


def build_plocgan_disc_input(
    img_3ch: torch.Tensor,
    protein_1ch: torch.Tensor,
    struct_ch_indices,
) -> torch.Tensor:
    """PLocGAN-style D input: (B,3,H,W) = [struct_ch0, protein_or_xt, struct_ch1]."""
    i0, i1 = int(struct_ch_indices[0]), int(struct_ch_indices[1])
    return torch.cat(
        [img_3ch[:, i0 : i0 + 1], protein_1ch, img_3ch[:, i1 : i1 + 1]],
        dim=1,
    )


class ImagePlocGANDiscriminator(nn.Module):
    """PLocGAN part2 pixel-domain discriminator (no DivCo contrast branch).

    Input ``[struct0, protein_or_xt, struct1]`` (B,3,H,W); 4 stride-2 Conv -> FC:
      - adv_logit: (B,1) scalar real/fake
      - y_aux: (B,2) sigmoid localization labels (MSE against soft labels)
    """

    def __init__(
        self,
        ndf: int = 32,
        aux_label_dim: int = 2,
        spatial_size: int = 256,
        in_ch: int = 3,
    ):
        super().__init__()
        self.ndf = int(ndf)
        self.aux_label_dim = int(aux_label_dim)
        self.in_ch = int(in_ch)
        sz = int(spatial_size)
        if sz != 256:
            raise ValueError(
                f"ImagePlocGANDiscriminator is fixed at 256->16 downsampling, got spatial_size={sz}"
            )

        self.conv1 = nn.Conv2d(self.in_ch, ndf, 4, 2, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(ndf, eps=1e-5, momentum=0.1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(ndf * 2, eps=1e-5, momentum=0.1)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(ndf * 4, eps=1e-5, momentum=0.1)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(ndf * 8, eps=1e-5, momentum=0.1)
        self.lrelu4 = nn.LeakyReLU(0.2)

        feat_dim = 16 * 16 * ndf * 8
        self.fc1 = nn.Linear(feat_dim, 256, bias=False)
        self.fc1_bn = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1)

        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 128, bias=False)
        self.fc3_bn = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.fc3_lrelu = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, self.aux_label_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.lrelu1(self.conv1_bn(self.conv1(x)))
        out = self.lrelu2(self.conv2_bn(self.conv2(out)))
        out = self.lrelu3(self.conv3_bn(self.conv3(out)))
        out = self.lrelu4(self.conv4_bn(self.conv4(out)))

        out = out.view(out.shape[0], -1)
        out = self.lrelu1(self.fc1_bn(self.fc1(out)))
        adv_logit = self.fc2(out)
        label_feat = self.fc3_lrelu(self.fc3_bn(self.fc3(out)))
        y_aux = torch.sigmoid(self.fc4(label_feat))
        return adv_logit, y_aux


def build_plocgan_latent_input(
    z_struct: torch.Tensor,
    protein_latent: torch.Tensor,
) -> torch.Tensor:
    """Latent-space PLocGAN D input: [z_struct, protein_latent].

    z_struct is the struct latent (nucleus + tubulin, 8ch for hpa_vae);
    protein_latent is the real z_protein or the predicted x̂₀ (4ch). Judging in the
    diffusion latent space avoids a VAE decode.
    """
    return torch.cat([z_struct, protein_latent], dim=1)


def _gn(num_ch: int) -> nn.GroupNorm:
    g = max(1, min(8, int(num_ch)))
    while int(num_ch) % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, int(num_ch))


def q_sample_posterior(
    diffusion,
    x_start: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample x_{t-1} ~ q(x_{t-1} | x_t, x_0); degenerates to x_0 at t==0."""
    mean, _, log_var = diffusion.q_posterior_mean_variance(x_start, x_t, t)
    if noise is None:
        noise = torch.randn_like(x_start)
    # t=0: the posterior is deterministic x_0
    nonzero = (t != 0).to(dtype=x_start.dtype).view(-1, *([1] * (x_start.ndim - 1)))
    return mean + nonzero * torch.exp(0.5 * log_var) * noise


class LatentLDDGANDiscriminator(nn.Module):
    """LDDGAN-style latent transition discriminator: D(x_{t-1}, x_t, t [, z_struct]).

    Judges whether stepping from x_t back to x_{t-1} looks like the true posterior,
    rather than scoring clean x_0 alone. GroupNorm avoids BatchNorm collapsing when
    a t-mask leaves N=1. The auxiliary head y_aux=(p_cyto, p_nucl) serves partial
    label candidate selection, not the main generative objective.
    """

    def __init__(
        self,
        protein_ch: int = 4,
        struct_ch: int = 8,
        spatial_size: int = 64,
        ndf: int = 64,
        use_struct: bool = True,
        aux_label_dim: int = 2,
    ):
        super().__init__()
        self.protein_ch = int(protein_ch)
        self.struct_ch = int(struct_ch)
        self.use_struct = bool(use_struct)
        self.ndf = int(ndf)
        self.aux_label_dim = int(aux_label_dim)
        # x_{t-1}(C) + x_t(C) + t_channel(1) [+ z_struct]
        self.in_ch = self.protein_ch * 2 + 1 + (self.struct_ch if self.use_struct else 0)
        sz = int(spatial_size)

        self.conv1 = nn.Conv2d(self.in_ch, ndf, 4, 2, 1, bias=False)
        self.norm1 = _gn(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.norm2 = _gn(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.norm3 = _gn(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.norm4 = _gn(ndf * 8)
        self.lrelu = nn.LeakyReLU(0.2)

        feat_sz = sz // 16
        if feat_sz < 1:
            raise ValueError(f"spatial_size={sz} is too small for four stride-2 downsampling stages")
        feat_dim = feat_sz * feat_sz * ndf * 8
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.aux_label_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_tm1: torch.Tensor,
        x_t: torch.Tensor,
        t_norm: torch.Tensor,
        z_struct: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x_tm1.shape
        t_ch = t_norm.reshape(b, 1, 1, 1).to(dtype=x_tm1.dtype).expand(b, 1, h, w)
        parts = [x_tm1, x_t, t_ch]
        if self.use_struct:
            if z_struct is None:
                raise ValueError("LatentLDDGANDiscriminator(use_struct=True) requires z_struct")
            parts.append(z_struct)
        x = torch.cat(parts, dim=1)
        out = self.lrelu(self.norm1(self.conv1(x)))
        out = self.lrelu(self.norm2(self.conv2(out)))
        out = self.lrelu(self.norm3(self.conv3(out)))
        out = self.lrelu(self.norm4(self.conv4(out)))
        out = out.flatten(1)
        feat = self.lrelu(self.fc1(out))
        adv_logit = self.fc2(feat)
        y_aux = torch.sigmoid(self.fc4(self.lrelu(self.fc3(feat))))
        return adv_logit, y_aux


class LatentPlocGANDiscriminator(nn.Module):
    """Latent-space PLocGAN discriminator on [z_struct, protein_latent] (B, C, h, w).

    Same layout as the pixel-domain ImagePlocGANDiscriminator (4 stride-2 Conv -> FC)
    but operating in the diffusion latent space (64×64 by default).
    """

    def __init__(
        self,
        in_ch: int,
        spatial_size: int = 64,
        ndf: int = 32,
        aux_label_dim: int = 2,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.ndf = int(ndf)
        self.aux_label_dim = int(aux_label_dim)
        sz = int(spatial_size)

        self.conv1 = nn.Conv2d(self.in_ch, ndf, 4, 2, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(ndf, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(ndf * 2, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(ndf * 4, eps=1e-5, momentum=0.1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(ndf * 8, eps=1e-5, momentum=0.1)
        self.lrelu = nn.LeakyReLU(0.2)

        feat_sz = sz // 16  # 4 stride-2 layers
        if feat_sz < 1:
            raise ValueError(f"spatial_size={sz} is too small for four stride-2 downsampling stages")
        feat_dim = feat_sz * feat_sz * ndf * 8
        self.fc1 = nn.Linear(feat_dim, 256, bias=False)
        self.fc1_bn = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1)
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 128, bias=False)
        self.fc3_bn = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.fc3_lrelu = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(128, self.aux_label_dim)

        self.apply(self._init_weights)

    @staticmethod
    def remap_legacy_state_dict(state_dict):
        """Kept for legacy ckpt key names; currently an identity mapping."""
        return state_dict

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.lrelu(self.conv1_bn(self.conv1(x)))
        out = self.lrelu(self.conv2_bn(self.conv2(out)))
        out = self.lrelu(self.conv3_bn(self.conv3(out)))
        out = self.lrelu(self.conv4_bn(self.conv4(out)))
        out = out.view(out.shape[0], -1)
        out = self.lrelu(self.fc1_bn(self.fc1(out)))
        adv_logit = self.fc2(out)
        label_feat = self.fc3_lrelu(self.fc3_bn(self.fc3(out)))
        y_aux = torch.sigmoid(self.fc4(label_feat))
        return adv_logit, y_aux


class ImagePatchDiscriminator(nn.Module):
    """PatchGAN-style pixel-domain discriminator on decoded protein maps (B, 1, H, W).

    Real/fake samples are chosen by the training script: intermediate x_t by
    default, with a clean/noisy endpoint fallback.
    """

    def __init__(
        self,
        in_ch: int = 1,
        spatial_size: int = 256,
        base_ch: int = 64,
        aux_label_dim: int = 2,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.aux_label_dim = int(aux_label_dim)
        self.backbone, c_feat = _patchgan_backbone(in_ch, spatial_size, base_ch)
        self.c_feat = c_feat
        self.adv_head = nn.Conv2d(c_feat, 1, kernel_size=4, stride=1, padding=1)
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_fc = nn.Linear(c_feat, self.aux_label_dim)
        nn.init.normal_(self.aux_fc.weight, std=0.02)
        nn.init.constant_(self.aux_fc.bias, 0.0)

    def forward(self, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x_img)
        adv_logits = self.adv_head(feat)
        y_aux = self.aux_fc(self.aux_pool(feat).flatten(1))
        return adv_logits, y_aux


def build_image_discriminator(
    args,
    *,
    spatial_size: int = 256,
):
    """Build the pixel-domain discriminator according to ``--latent-adv-mode``."""
    mode = str(getattr(args, "latent_adv_mode", "plocgan") or "plocgan").lower()
    base_ch = int(getattr(args, "latent_adv_base_ch", 64))
    add_t = bool(getattr(args, "latent_adv_noise_t_channel", False))
    if mode == "plocgan":
        return ImagePlocGANDiscriminator(
            ndf=base_ch,
            spatial_size=spatial_size,
            in_ch=4 if add_t else 3,
        )
    if mode == "simple":
        return ImagePatchDiscriminator(
            in_ch=2 if add_t else 1,
            spatial_size=spatial_size,
            base_ch=base_ch,
        )
    raise ValueError(f"--latent-adv-mode supports plocgan (recommended) / simple, got: {mode!r}")


def build_latent_discriminator(
    args,
    *,
    in_ch: int,
    spatial_size: int,
    protein_ch: int = 4,
    struct_ch: int = 8,
):
    """Build the latent-space discriminator.

    - ``lddgan``: D(x_{t-1}, x_t, t, z_struct), intermediate transition judging
      (recommended: fewer steps, higher quality)
    - ``plocgan``: D([z_struct, x̂₀]), endpoint judging plus an optional label head
    """
    mode = str(getattr(args, "latent_adv_mode", "lddgan") or "lddgan").lower()
    base_ch = int(getattr(args, "latent_adv_base_ch", 64))
    if mode == "lddgan":
        return LatentLDDGANDiscriminator(
            protein_ch=int(protein_ch),
            struct_ch=int(struct_ch),
            spatial_size=int(spatial_size),
            ndf=base_ch,
            use_struct=True,
            aux_label_dim=2,
        )
    if mode == "plocgan":
        return LatentPlocGANDiscriminator(
            in_ch=int(in_ch),
            spatial_size=int(spatial_size),
            ndf=base_ch,
        )
    raise ValueError(
        f"--latent-adv-mode supports lddgan / plocgan, got: {mode!r}"
    )


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    r = real_logits.flatten(1).mean(dim=1)
    f = fake_logits.flatten(1).mean(dim=1)
    return F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    f = fake_logits.flatten(1).mean(dim=1)
    return (-f).mean()


def plocgan_discriminator_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Non-saturating discriminator loss using softplus.

    Avoids gradient vanishing in latent space where real/fake separation is easy.
    Softplus(-x) = log(1 + e^(-x)) provides smoother gradients than BCE.
    """
    r = real_logits.view(-1)
    f = fake_logits.view(-1)
    # Real samples should have large positive logits
    loss_real = F.softplus(-r).mean()
    # Fake samples should have large negative logits
    loss_fake = F.softplus(f).mean()
    return loss_real + loss_fake


def plocgan_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: min -log(D(G(z))) instead of max log(1-D(G(z))).

    This provides stronger gradients when D is strong, avoiding the saturating
    regime of vanilla GAN where G gets no gradient when D is confident.
    """
    f = fake_logits.view(-1)
    # G wants D(fake) to be large (fooling the discriminator)
    return F.softplus(-f).mean()


def r1_gradient_penalty(
    disc: nn.Module,
    real_input: torch.Tensor,
    lamb: float = 10.0,
) -> torch.Tensor:
    """R1 gradient penalty: ‖∇_x D(x)‖² on real samples.

    Keeps D from overfitting the real data manifold and stabilises hinge GAN training.
    """
    real_input_grad = real_input.detach().requires_grad_(True)
    logits, _ = disc(real_input_grad)
    real_pred = logits.flatten(1).mean(dim=1).sum()
    grad_real = torch.autograd.grad(
        outputs=real_pred,
        inputs=real_input_grad,
        create_graph=True,
        only_inputs=True,
    )[0]
    gp = grad_real.pow(2).flatten(1).sum(dim=1).mean()
    return lamb * gp


def image_disc_forward(
    disc: nn.Module,
    x_img: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Unified pixel-domain D forward; returns (logits, y_aux)."""
    return disc(x_img)
