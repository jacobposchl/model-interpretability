"""
Circuit analysis utilities for Phase 1 meta-encoder validation.

Handles data collection from the frozen backbone + meta-encoder, pairwise
alignment profile computation, and class purity analysis for discovered
circuit clusters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
_STD  = torch.tensor([0.2470, 0.2435, 0.2616])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalised CIFAR-10 tensor -> [0, 1]. Accepts [C,H,W] or [B,C,H,W]."""
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    if x.dim() == 4:
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    else:
        mean = mean[:, None, None]
        std  = std[:, None, None]
    return (x * std + mean).clamp(0, 1)


class CircuitAnalyzer:
    """
    Collects representations from the frozen backbone + trained meta-encoder
    and computes alignment profiles for downstream circuit discovery.
    """

    def __init__(
        self,
        backbone,
        meta_encoder,
        loader: DataLoader,
        device: torch.device,
    ):
        self.backbone = backbone
        self.meta_encoder = meta_encoder
        self.loader = loader
        self.device = device

    @torch.no_grad()
    def collect_representations(self, max_samples: int = 10000) -> dict:
        """
        Collect trajectories, per-layer z-vectors, images, and labels.

        Returns dict with:
            trajectories: list of L tensors, each [N, D_l] (CPU)
            z_list:       list of L tensors, each [N, d] (CPU)
            labels:       [N] integer class labels (CPU)
            images:       [N, 3, 32, 32] normalized images (CPU)
        """
        self.meta_encoder.eval()

        all_trajs: list[list] | None = None
        all_z: list[list] | None = None
        all_labels = []
        all_images = []
        n = 0

        for batch in self.loader:
            images = batch[0].to(self.device)
            labels = batch[-1]

            trajectory = self.backbone(images)
            z_list = self.meta_encoder(trajectory)

            if all_trajs is None:
                all_trajs = [[] for _ in range(len(trajectory))]
                all_z = [[] for _ in range(len(z_list))]

            for l, h in enumerate(trajectory):
                all_trajs[l].append(h.cpu())
            for l, z in enumerate(z_list):
                all_z[l].append(z.cpu())

            all_labels.append(labels.cpu())
            all_images.append(images.cpu())

            n += images.shape[0]
            if n >= max_samples:
                break

        L = len(all_trajs)
        trajectories = [torch.cat(all_trajs[l], 0)[:max_samples] for l in range(L)]
        z_list = [torch.cat(all_z[l], 0)[:max_samples] for l in range(L)]
        labels = torch.cat(all_labels, 0)[:max_samples]
        images = torch.cat(all_images, 0)[:max_samples]

        return {
            "trajectories": trajectories,
            "z_list": z_list,
            "labels": labels,
            "images": images,
        }

    @staticmethod
    def compute_all_profiles(
        trajectories: list[torch.Tensor],
        chunk_size: int = 1000,
    ) -> torch.Tensor:
        """
        Compute pairwise alignment profiles from L2-normalized trajectories.

        For large N, computes in chunks to avoid OOM.

        Args:
            trajectories: list of L tensors, each [N, D_l], L2-normalized
            chunk_size:   max rows to process at once

        Returns:
            [N, N, L] pairwise per-layer cosine similarities
        """
        L = len(trajectories)
        N = trajectories[0].shape[0]

        profiles = torch.zeros(N, N, L)

        for l in range(L):
            h = trajectories[l]  # [N, D_l]
            # Compute full pairwise similarity in chunks
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                profiles[i:end, :, l] = h[i:end] @ h.t()

        return profiles

    @staticmethod
    def compute_pair_profiles(
        trajectories: list[torch.Tensor],
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment profiles for specific pairs.

        Args:
            trajectories: list of L tensors, each [N, D_l], L2-normalized
            idx_a, idx_b: [N_pairs] indices into trajectories

        Returns:
            [N_pairs, L] per-layer cosine similarities
        """
        L = len(trajectories)
        N_pairs = idx_a.shape[0]
        profiles = torch.zeros(N_pairs, L)

        for l in range(L):
            h_a = trajectories[l][idx_a]  # [N_pairs, D_l]
            h_b = trajectories[l][idx_b]
            profiles[:, l] = (h_a * h_b).sum(dim=-1)

        return profiles

    @staticmethod
    def compute_class_purity(
        pair_indices: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Compute class purity for a subset of pairs.

        Args:
            pair_indices: [N_pairs, 2] input indices per pair
            labels:       [N] class labels
            mask:         [N_pairs] boolean mask selecting cluster pairs

        Returns:
            Purity score in [0, 1]
        """
        selected = pair_indices[mask]
        unique_inputs = selected.unique()
        input_labels = labels[unique_inputs]

        if len(input_labels) == 0:
            return 0.0

        counts = input_labels.bincount()
        return float(counts.max()) / len(input_labels)
