"""
Fidelity loss L_info: trains per-layer z-representations to contain sufficient
information to reconstruct the corresponding alignment profile entries.

L_info = (1/L) * sum_l || MLP(z_l^a * z_l^b) - s_l(a, b) ||^2

The element-wise product z_l^a * z_l^b is symmetric by construction (swapping
a and b gives the same vector), which matches the symmetry of cosine similarity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.meta_encoder import ProfileRegressor


class InfoLoss(nn.Module):
    """
    Profile reconstruction fidelity loss.

    For each layer l, computes the element-wise product of the two inputs'
    z-vectors, passes it through the ProfileRegressor MLP, and measures
    MSE against the ground-truth per-layer cosine similarity.
    """

    def __init__(self, regressor: ProfileRegressor):
        super().__init__()
        self.regressor = regressor

    def forward(
        self,
        z_list_a: list[torch.Tensor],
        z_list_b: list[torch.Tensor],
        true_similarities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_list_a: list of L tensors, each [N_pairs, d]
            z_list_b: list of L tensors, each [N_pairs, d]
            true_similarities: [N_pairs, L] ground-truth per-layer cosine sims

        Returns:
            Scalar loss (mean over layers and pairs).
        """
        L = len(z_list_a)
        total_loss = 0.0

        for l in range(L):
            # Element-wise product (symmetric combination)
            z_product = z_list_a[l] * z_list_b[l]  # [N_pairs, d]

            # Predict similarity
            predicted = self.regressor(z_product)  # [N_pairs]

            # MSE against ground truth for this layer
            target = true_similarities[:, l]  # [N_pairs]
            total_loss = total_loss + torch.mean((predicted - target) ** 2)

        return total_loss / L
