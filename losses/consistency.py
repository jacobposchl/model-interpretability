"""
Depth-weighted circuit consistency loss.

ℒ_cons(x₁, x₂) = Σ_l  w_l · (1 - cos(h_l(x₁), h_l(x₂)))

Applied to pairs of same-category inputs. Penalises divergence between their
per-layer circuit representations, with the penalty controlled per layer by
a weight scheme. Uses cosine distance, which is dimension-independent and
ranges [0, 1] per sample regardless of activation dimensionality.

Weight schemes:
  'linear'          — w_l = l / Σl            (ramp up; late layers penalised more)
  'exponential'     — w_l = exp(l) / Σexp     (sharp late-layer emphasis)
  'uniform'         — w_l = 1/L               (ablation baseline)
  'inverted_linear' — w_l = (L-l+1) / Σ(...)  (ramp down; early layers penalised more)
  'early_only'      — w_l = 1/pivot if l ≤ pivot, else 0  (requires pivot_layer > 0)

Rationale for 'early_only' / 'inverted_linear':
  Early layers capture coarse, shared structure (edges, textures, shapes) that
  should be reused across semantically similar classes. Later layers form
  class-specific discriminative features — constraining those for novel inputs
  prevents the model from building fresh representations beyond the training
  distribution. Concentrating circuit consistency pressure on early layers
  preserves shared low-level structure while leaving later layers free to
  diverge for novel classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _depth_weights(
    num_layers: int,
    scheme: str,
    device: torch.device,
    pivot_layer: int = 0,
) -> torch.Tensor:
    if scheme == "linear":
        w = torch.arange(1, num_layers + 1, dtype=torch.float, device=device)
    elif scheme == "exponential":
        w = torch.exp(torch.linspace(0.0, 2.0, num_layers, device=device))
    elif scheme == "uniform":
        w = torch.ones(num_layers, dtype=torch.float, device=device)
    elif scheme == "inverted_linear":
        w = torch.arange(num_layers, 0, -1, dtype=torch.float, device=device)
    elif scheme == "early_only":
        if pivot_layer <= 0 or pivot_layer > num_layers:
            raise ValueError(
                f"pivot_layer must be in [1, {num_layers}] for early_only, got {pivot_layer}"
            )
        w = torch.zeros(num_layers, dtype=torch.float, device=device)
        w[:pivot_layer] = 1.0
    else:
        raise ValueError(
            f"Unknown weight scheme: '{scheme}'. "
            "Use linear/exponential/uniform/inverted_linear/early_only."
        )
    return w / w.sum()


class CircuitConsistencyLoss(nn.Module):
    def __init__(self, weight_scheme: str = "linear", pivot_layer: int = 0):
        super().__init__()
        self.weight_scheme = weight_scheme
        self.pivot_layer = pivot_layer
        self._cached_weights: torch.Tensor | None = None
        self._cached_L: int = 0

    def _get_weights(self, num_layers: int, device: torch.device) -> torch.Tensor:
        if self._cached_weights is None or self._cached_L != num_layers:
            self._cached_weights = _depth_weights(
                num_layers, self.weight_scheme, device, self.pivot_layer
            )
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
