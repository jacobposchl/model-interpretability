"""
Meta-encoder E: trajectory → circuit embedding z.

Takes the full activation trajectory T(x) = (h₁, h₂, ..., h_L) — a list of
per-layer representations — and compresses it into a compact vector z that
lives in the circuit latent space.

Key design choices:
  - Each layer is projected to a common hidden dimension first, since layers
    have different widths (e.g. [64, 64, 128, 128, 256, 256, 512, 512] for
    ResNet18). This makes the encoder architecture-agnostic.
  - Two encoder variants:
      'mlp'         — Concatenate all projected layers, pass through 2-layer MLP.
                      Fast, simple, works well when L is small.
      'transformer' — Stack projected layers as a sequence with positional
                      encodings, run a small 2-layer transformer, pool.
                      Better when L is large or layer ordering matters.
  - Output z has L2 normalization applied before returning, so distances in
    the circuit latent space are cosine distances. This stabilises the
    consistency loss and makes UMAP visualizations more meaningful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MetaEncoder(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        encoder_type: str = "mlp",
    ):
        super().__init__()
        self.layer_dims = layer_dims
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        L = len(layer_dims)

        # Per-layer projection to a common hidden_dim
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for d in layer_dims
        ])

        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(hidden_dim * L, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            )

        elif encoder_type == "transformer":
            # Positional encoding (fixed sinusoidal)
            self.register_buffer("pos_enc", _sinusoidal_pos_enc(L, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.readout = nn.Linear(hidden_dim, embedding_dim)

        else:
            raise ValueError(f"encoder_type must be 'mlp' or 'transformer', got '{encoder_type}'")

    def forward(self, trajectory: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            trajectory: list of L tensors, each shape [B, D_l]

        Returns:
            z: circuit embedding, shape [B, embedding_dim], L2-normalised
        """
        assert len(trajectory) == len(self.projectors), (
            f"Trajectory has {len(trajectory)} layers but encoder expects {len(self.projectors)}"
        )

        projected = [proj(h) for proj, h in zip(self.projectors, trajectory)]

        if self.encoder_type == "mlp":
            cat = torch.cat(projected, dim=-1)        # [B, hidden_dim * L]
            z = self.encoder(cat)                     # [B, embedding_dim]

        else:  # transformer
            stacked = torch.stack(projected, dim=1)   # [B, L, hidden_dim]
            stacked = stacked + self.pos_enc           # add positional encoding
            encoded = self.transformer(stacked)       # [B, L, hidden_dim]
            pooled = encoded.mean(dim=1)              # [B, hidden_dim]
            z = self.readout(pooled)                  # [B, embedding_dim]

        return F.normalize(z, dim=-1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sinusoidal_pos_enc(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape [1, length, dim]."""
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, L, D]
