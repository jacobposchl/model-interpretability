"""
NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss.

Used in Stage 1 to train the meta-encoder in isolation, before the full
CTLS consistency loss is introduced. This lets the meta-encoder learn a
meaningful circuit latent space using a standard SimCLR-style objective,
establishing a baseline for how much semantic structure exists in the circuit
space of a normally-trained model.

Reference: Chen et al. "A Simple Framework for Contrastive Learning" (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: circuit embeddings for first view, shape [B, D]
            z2: circuit embeddings for second view, shape [B, D]

        Returns:
            Scalar NT-Xent loss.
        """
        B = z1.shape[0]

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: [2B, D]
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: [2B, 2B]
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        return F.cross_entropy(sim, labels)
