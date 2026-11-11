"""Image quality metrics aligned with the ProLDM / PLocGAN papers.

Fidelity: PSNR, SSIM (pixel space [0,1]).
Diversity: LPIPS (pairwise VGG perceptual distance between samples; higher = more diverse).
Distribution: FID (Inception-2048) and Density / Coverage (VGG16 classifier[:4], 4096-d, as in ProLDM).
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.spatial.distance
import torch
from torchvision import models, transforms


def compute_density_coverage(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    nearest_k: int = 10,
) -> tuple[float, float]:
    """Density / Coverage, identical to ProLDM ``evaluate/prdc.py::compute_prdc``.

    Eqs. 2-9 / 2-11: the k-NN radius is the k-th nearest neighbour distance
    within the real set, excluding self.
    """
    real_features = np.asarray(real_features, dtype=np.float32)
    fake_features = np.asarray(fake_features, dtype=np.float32)
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("real_features / fake_features must be 2-D arrays")
    n_real = real_features.shape[0]
    if n_real <= nearest_k:
        raise ValueError(
            f"real sample count ({n_real}) must be > nearest_k ({nearest_k})"
        )
    if fake_features.shape[0] == 0:
        raise ValueError("fake_features is empty")

    real_nn_dist = _nearest_neighbour_distances(real_features, nearest_k)
    dist_rf = scipy.spatial.distance.cdist(real_features, fake_features, metric="euclidean")

    density = (1.0 / float(nearest_k)) * (
        dist_rf < np.expand_dims(real_nn_dist, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        dist_rf.min(axis=1) < real_nn_dist
    ).mean()

    return float(density), float(coverage)


def _nearest_neighbour_distances(input_features: np.ndarray, nearest_k: int) -> np.ndarray:
    """ProLDM prdc: k-th nearest neighbour distance (self-distance is 0, so take k+1)."""
    dists = scipy.spatial.distance.cdist(
        input_features, input_features, metric="euclidean"
    )
    return _get_kth_value(dists, k=nearest_k + 1, axis=-1)


def _get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


class VGG16FeatureAccumulator:
    """Batched version of ProLDM ``evaluate_model.extract_features``.

    Single-channel protein image in [0,1] -> resize 224 -> replicate to 3 channels
    -> VGG16 features -> avgpool -> flatten -> classifier[:4], giving 4096-d.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._to_tensor = transforms.ToTensor()
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            self.net = models.vgg16(weights=weights).to(device).eval()
        except AttributeError:
            self.net = models.vgg16(pretrained=True).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self._real: list[np.ndarray] = []
        self._fake: list[np.ndarray] = []

    @torch.no_grad()
    def _extract_batch(self, x_protein: torch.Tensor) -> np.ndarray:
        """x_protein: (B,1,H,W) in [-1,1]."""
        x01 = ((x_protein.float() + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()
        feats: list[np.ndarray] = []
        for b in range(x01.shape[0]):
            img = x01[b, 0]
            img224 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            img3d = np.reshape(img224, (1, 224, 224)).repeat(3, axis=0)
            inp = self._to_tensor(img3d).reshape(1, 3, 224, 224).float().to(self.device)
            ifeatures = self.net.features(inp)
            ifeatures = self.net.avgpool(ifeatures)
            ifeatures = torch.flatten(ifeatures, 1)
            feat = self.net.classifier[:4](ifeatures).cpu().numpy()
            feats.append(feat)
        return np.concatenate(feats, axis=0).astype(np.float32)

    def update(self, x_protein_batch: torch.Tensor, real: bool):
        feats = self._extract_batch(x_protein_batch)
        (self._real if real else self._fake).append(feats)

    def stacked_features(self, real: bool) -> np.ndarray:
        feats_list = self._real if real else self._fake
        if not feats_list:
            raise ValueError("no VGG16 features accumulated")
        return np.concatenate(feats_list, axis=0).astype(np.float32)


def protein_to_lpips_rgb(x_protein: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) in [-1,1] -> (B,3,H,W) in [-1,1] (replicated for LPIPS)."""
    if x_protein.dim() != 4 or x_protein.shape[1] != 1:
        raise ValueError(f"expected (B,1,H,W), got {tuple(x_protein.shape)}")
    return x_protein.repeat(1, 3, 1, 1)


class LPIPSDiversityEvaluator:
    """Mean pairwise LPIPS over multiple samples drawn from the same condition."""

    def __init__(self, device: torch.device, net: str = "vgg"):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS requires the lpips package: pip install lpips"
            ) from exc
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def pairwise_distance(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        """img_a, img_b: (1,H,W) in [-1,1] on CPU."""
        a = protein_to_lpips_rgb(img_a.unsqueeze(0)).to(self.device)
        b = protein_to_lpips_rgb(img_b.unsqueeze(0)).to(self.device)
        return float(self.loss_fn(a, b).item())

    @torch.no_grad()
    def mean_pairwise_diversity(
        self,
        samples_by_idx: dict[int, list[torch.Tensor]],
    ) -> tuple[float, float, int]:
        """All C(n,2) pairs per idx; returns (mean, std, n_pairs)."""
        all_vals: list[float] = []
        for _idx, imgs in samples_by_idx.items():
            n = len(imgs)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    all_vals.append(self.pairwise_distance(imgs[i], imgs[j]))
        if not all_vals:
            return float("nan"), float("nan"), 0
        arr = np.asarray(all_vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std()), int(arr.size)
