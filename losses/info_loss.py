"""
Fidelity loss L_info: trains per-layer z-representations to encode the flow
co-activation structure of each backbone block.

L_info = (1/L) * sum_l || MLP_l(z_l^a * z_l^b) - (f_l^a ⊙ f_l^b) ||^2

The target f_l^a ⊙ f_l^b is the element-wise product of the two compressed,
L2-normalized flow vectors at layer l.  f_l(x) is derived from the non-skip
branch output F_l(x) = bn2(x) (pre-residual addition), compressed via
AdaptiveMaxPool2d + Flatten + a fixed linear projection to D_flow dimensions.
This isolates each block's contribution from the accumulated residual history,
making it the correct signal for circuit detection.

With the flow compression, all layers share the same D_flow output dimension,
so layer_dims = [D_flow] * L and the regressor architecture is uniform.
The element-wise product z_l^a * z_l^b is symmetric, matching the symmetry
of the co-activation target f_l^a ⊙ f_l^b.
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
            layer_dims:     Per-layer flow dimensions (= [D_flow] * L with the
                            flow-compression backbone).  One regressor per layer
                            with output_dim = layer_dims[l].
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
            rich_targets: list of L tensors, each [N_pairs, D_flow] — the flow
                          co-activation vectors (f_l^a ⊙ f_l^b)

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
