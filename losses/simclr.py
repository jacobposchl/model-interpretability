"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss — SimCLR.

For a batch of N samples, each with two augmented views (z1, z2):
  - The positive pair for sample i is (z1_i, z2_i).
  - All other 2(N-1) embeddings in the batch are negatives.

The loss is symmetric: it is computed for both z1→z2 and z2→z1 directions
and averaged. This is the standard SimCLR formulation from Chen et al. (2020).

Key differences from SupConLoss (losses/contrastive.py):
  - No class labels: positives are defined by augmentation pairing, not category.
  - Each anchor has exactly one positive (its augmented counterpart), not multiple
    same-class anchors. This means there is no per-class averaging.

Reference: Chen et al. "A Simple Framework for Contrastive Learning of Visual
Representations" (ICML 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: [N, D] L2-normalised embeddings from view 1
            z2: [N, D] L2-normalised embeddings from view 2

        Returns:
            Scalar NT-Xent loss.
        """
        N = z1.shape[0]
        device = z1.device

        # Concatenate views: [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarity scaled by temperature: [2N, 2N]
        sim = torch.mm(z, z.t()) / self.temperature

        # Numerically stable: subtract row-wise max
        sim = sim - sim.detach().max(dim=1, keepdim=True).values

        # Positive pair indices: (i, i+N) and (i+N, i)
        pos_i = torch.arange(N, device=device)
        pos_j = torch.arange(N, 2 * N, device=device)

        # Mask out self-similarity on the diagonal
        diag_mask = torch.eye(2 * N, device=device, dtype=torch.bool)

        exp_sim = torch.exp(sim).masked_fill(diag_mask, 0.0)   # [2N, 2N]
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)       # [2N]

        # Loss for each direction: z1→z2 and z2→z1
        loss_12 = -(sim[pos_i, pos_j] - log_denom[pos_i]).mean()
        loss_21 = -(sim[pos_j, pos_i] - log_denom[pos_j]).mean()

        return (loss_12 + loss_21) / 2.0
