"""
Geometry loss L_geometry: enforces that inputs with similar alignment profiles
are geometrically close in z-space, using a soft contrastive objective.

L_geometry = (1/L) * sum_l [ -sum_{a,b} P_bar_l(a,b) * log_softmax(sim/tau) ]

where P_bar_l(a,b) is the per-layer true similarity normalized across all
pairs in the batch to form a soft target distribution, and tau is a temperature.

All z vectors are L2-normalized before similarity computation. The bounded
structure of the unit hypersphere provides implicit geometric repulsion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryLoss(nn.Module):
    """
    Soft contrastive geometry loss using profile-derived target distributions.

    For each layer l:
    1. Compute pairwise cosine similarity in z-space: sim_l[a,b] = z_l^a . z_l^b
    2. Form soft targets from true profile similarities (per-row normalization)
    3. Cross-entropy between soft targets and log-softmax of sim/tau
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_list: list[torch.Tensor],
        true_similarities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_list: list of L tensors, each [B, d] (L2-normalized)
            true_similarities: [B, B, L] pairwise per-layer cosine similarities

        Returns:
            Scalar loss (mean over layers).
        """
        B = z_list[0].shape[0]
        L = len(z_list)

        # Mask to exclude self-pairs (diagonal)
        mask = ~torch.eye(B, dtype=torch.bool, device=z_list[0].device)  # [B, B]

        total_loss = 0.0

        for l in range(L):
            z_l = z_list[l]  # [B, d], already L2-normalized

            # Pairwise cosine similarity in z-space
            sim = z_l @ z_l.t()  # [B, B]

            # True similarities for this layer
            true_sim_l = true_similarities[:, :, l]  # [B, B]

            # Form soft targets: normalize true similarities per row (excluding self)
            # Clamp to non-negative to ensure valid distribution
            targets = true_sim_l.clone()
            targets[~mask] = 0.0  # zero out diagonal
            targets = targets.clamp(min=0.0)  # ensure non-negative

            # Per-row normalization to form probability distribution
            row_sums = targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
            targets = targets / row_sums  # [B, B], rows sum to 1

            # Log-softmax of sim/tau (excluding self from denominator)
            logits = sim / self.temperature
            logits[~mask] = float("-inf")  # mask self-pairs
            log_probs = F.log_softmax(logits, dim=1)  # [B, B]

            # Cross-entropy: -sum_b P_bar(a,b) * log_softmax(sim(a,b)/tau)
            # nan_to_num handles 0 * -inf = 0 (masked self-pairs with zero target)
            loss_l = -(torch.nan_to_num(targets * log_probs, nan=0.0)).sum(dim=1).mean()

            total_loss = total_loss + loss_l

        return total_loss / L
