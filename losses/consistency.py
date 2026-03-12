"""
Depth-weighted circuit consistency loss.

ℒ_cons(x₁, x₂) = Σ_l  w_l · (1 - cos(h_l(x₁), h_l(x₂)))

Applied to pairs of same-category inputs. Penalises divergence between their
per-layer circuit representations, with the penalty growing with layer depth.
Uses cosine distance, which is dimension-independent and ranges [0, 1] per
sample regardless of activation dimensionality.

Depth weighting rationale:
  Early layers extract surface features (edges, textures) that legitimately
  vary across instances of the same category — two dogs at different scales
  will have genuinely different early-layer responses. Later layers encode
  abstract semantic content that should be consistent for same-category
  inputs regardless of surface variation.

  Three weight schemes are provided:
    'linear'      — w_l = l / Σl         (gradual ramp)
    'exponential' — w_l = exp(l) / Σexp  (sharp late-layer emphasis)
    'uniform'     — w_l = 1/L            (ablation baseline, Stage 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _depth_weights(num_layers: int, scheme: str, device: torch.device) -> torch.Tensor:
    if scheme == "linear":
        w = torch.arange(1, num_layers + 1, dtype=torch.float, device=device)
    elif scheme == "exponential":
        w = torch.exp(torch.linspace(0.0, 2.0, num_layers, device=device))
    elif scheme == "uniform":
        w = torch.ones(num_layers, dtype=torch.float, device=device)
    else:
        raise ValueError(f"Unknown weight scheme: '{scheme}'. Use linear/exponential/uniform.")
    return w / w.sum()


class CircuitConsistencyLoss(nn.Module):
    def __init__(self, weight_scheme: str = "linear"):
        super().__init__()
        self.weight_scheme = weight_scheme
        self._cached_weights: torch.Tensor | None = None
        self._cached_L: int = 0

    def _get_weights(self, num_layers: int, device: torch.device) -> torch.Tensor:
        if self._cached_weights is None or self._cached_L != num_layers:
            self._cached_weights = _depth_weights(num_layers, self.weight_scheme, device)
            self._cached_L = num_layers
        return self._cached_weights.to(device)

    def forward(
        self,
        traj1: list[torch.Tensor],
        traj2: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            traj1: list of L tensors, each [B, D_l] — trajectory for first image
            traj2: list of L tensors, each [B, D_l] — trajectory for second image
                   (must be same-category as traj1)

        Returns:
            Scalar consistency loss.
        """
        assert len(traj1) == len(traj2), (
            f"Trajectory length mismatch: {len(traj1)} vs {len(traj2)}"
        )
        L = len(traj1)
        weights = self._get_weights(L, traj1[0].device)

        loss = torch.zeros(1, device=traj1[0].device)
        for h1, h2, w in zip(traj1, traj2, weights):
            d = (1.0 - F.cosine_similarity(h1, h2, dim=-1)).mean()
            loss = loss + w * d

        return loss.squeeze()
