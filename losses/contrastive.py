"""
Supervised Contrastive Loss (SupCon) for circuit embeddings.

Operates on the meta-encoder output z ∈ R^D for a batch of 2B samples
(B from x1, B from x2). Uses ground-truth class labels to define positives
and negatives, preventing the degenerate collapse that a purely attractive
consistency loss causes.

For each anchor i:
  Positives P(i) — all j ≠ i with the same class label
  Negatives      — all j ≠ i with a different class label (implicit via denominator)

  L_i = -1/|P(i)| Σ_{p ∈ P(i)} log [ exp(z_i · z_p / τ) / Σ_{j ≠ i} exp(z_i · z_j / τ) ]

With a batch of 2B = 256 samples across 10 classes, each anchor has ~25 positives
and ~230 negatives, giving a strong separation signal.

Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      circuit embeddings, shape [N, D] (need not be pre-normalised)
            labels: class labels, shape [N]

        Returns:
            Scalar supervised contrastive loss.
        """
        N = z.shape[0]
        z = F.normalize(z, dim=-1)

        # Pairwise cosine similarity scaled by temperature: [N, N]
        sim = torch.mm(z, z.t()) / self.temperature

        diag_mask = torch.eye(N, device=z.device, dtype=torch.bool)

        # Positive mask: same label, excluding self
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~diag_mask  # [N, N]

        # Numerically stable softmax: subtract row-wise max
        sim = sim - sim.detach().max(dim=1, keepdim=True).values

        # Denominator: sum exp(sim) over all j ≠ i
        exp_sim = torch.exp(sim).masked_fill(diag_mask, 0.0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [N, 1]

        # Log-probability for every (anchor, candidate) pair
        log_prob = sim - log_denom  # [N, N]

        # Average log-prob over positives; skip anchors with no positives
        n_pos = pos_mask.sum(dim=1).float()  # [N]
        valid = n_pos > 0
        pos_log_prob_sum = (pos_mask.float() * log_prob).sum(dim=1)  # [N]
        loss = -(pos_log_prob_sum[valid] / n_pos[valid]).mean()
        return loss
