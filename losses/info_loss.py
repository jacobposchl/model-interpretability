"""
Fidelity loss L_info: trains per-layer z-representations to encode the rich
per-channel co-activation structure of the backbone's alignment profiles.

L_info = (1/L) * sum_l || MLP_l(z_l^a * z_l^b) - (norm(h_l^a) ⊙ norm(h_l^b)) ||^2

The target is the per-channel co-activation vector at each layer — the element-
wise product of the two L2-normalized (flattened, un-pooled) activations. This
strictly generalises the old scalar cosine similarity: summing over channels
recovers the original dot product, but the per-channel vector preserves which
channels the two inputs co-activate, not just how much.

Each layer has its own ProfileRegressor with output_dim = D_l (the flattened
spatial dimension of that backbone layer), since D_l varies across layers.
The element-wise product z_l^a * z_l^b is symmetric by construction, matching
the symmetry of the rich profile target.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.meta_encoder import ProfileRegressor


class InfoLoss(nn.Module):
    """
    Profile reconstruction fidelity loss.

    Owns one ProfileRegressor per backbone layer. For each layer l, computes
    the element-wise product of the two inputs' z-vectors, passes it through
    the layer-specific regressor, and measures MSE against the ground-truth
    per-channel co-activation vector.
    """

    def __init__(
        self,
        layer_dims: list[int],
        projection_dim: int,
        hidden_dim: int,
    ):
        """
        Args:
            layer_dims:     List of flattened backbone dimensions per layer
                            (D_l = C_l * H_l * W_l). One regressor is built
                            per layer with output_dim = D_l.
            projection_dim: Input dimension to each regressor (= z vector dim d).
            hidden_dim:     Hidden dimension of each regressor MLP.
        """
        super().__init__()
        self.regressors = nn.ModuleList([
            ProfileRegressor(
                input_dim=projection_dim,
                hidden_dim=hidden_dim,
                output_dim=D_l,
            )
            for D_l in layer_dims
        ])

    def forward(
        self,
        z_list_a: list[torch.Tensor],
        z_list_b: list[torch.Tensor],
        rich_targets: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            z_list_a:     list of L tensors, each [N_pairs, d]
            z_list_b:     list of L tensors, each [N_pairs, d]
            rich_targets: list of L tensors, each [N_pairs, D_l] — the per-channel
                          co-activation vectors (norm(h_l^a) ⊙ norm(h_l^b))

        Returns:
            Scalar loss (mean MSE over layers).
        """
        L = len(z_list_a)
        total_loss = 0.0

        for l in range(L):
            z_product = z_list_a[l] * z_list_b[l]          # [N_pairs, d]
            predicted = self.regressors[l](z_product)       # [N_pairs, D_l]
            total_loss = total_loss + torch.mean((predicted - rich_targets[l]) ** 2)

        return total_loss / L
