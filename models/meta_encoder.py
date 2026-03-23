"""
Meta-encoder E: trajectory → circuit embedding z.

Takes the full activation trajectory T(x) = (h₁, h₂, ..., h_L) — a list of
per-layer representations — and compresses it into a compact vector z that
lives in the circuit latent space.

Key design choices:
  - Each layer is projected to a common projection_dim first, since layers have
    different widths (e.g. [64, 64, 128, 128, 256, 256, 512, 512] for
    ResNet18). This makes the encoder architecture-agnostic.
  - Two encoder variants:
      'weighted_sum'    — Project each layer to projection_dim d, then compute
                          z = Σ_l w_l · p_l where w_l = l / Σ(1..L) (linear
                          depth ramp). Depth weighting is a geometric property
                          of the representation, not an external loss coefficient.
      'transformer_cls' — Project each layer to projection_dim d, prepend a
                          learnable CLS token, add sinusoidal depth-encoding
                          positional embeddings, run a 2-layer transformer,
                          extract CLS token. Allows the model to learn which
                          layers matter per input.
                          Requires projection_dim divisible by 4 (4 attn heads).
  - Output z has L2 normalization applied before returning, so distances in
    the circuit latent space are cosine distances.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaEncoder(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        embedding_dim: int = 64,
        encoder_type: str = "weighted_sum",
        projection_dim: int = 128,
    ):
        """
        Args:
            layer_dims:     List of activation dimensions per layer (from backbone).
            embedding_dim:  Dimension of output circuit embedding z.
            encoder_type:   One of 'weighted_sum', 'transformer_cls'.
            projection_dim: Per-layer compression width. Must be divisible by 4
                            for 'transformer_cls' (4 attention heads).
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        L = len(layer_dims)

        valid_types = ("weighted_sum", "transformer_cls")
        if encoder_type not in valid_types:
            raise ValueError(
                f"encoder_type must be one of {valid_types}, got '{encoder_type}'"
            )

        # Per-layer projection to a common projection_dim
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
            )
            for d in layer_dims
        ])

        if encoder_type == "weighted_sum":
            # Fixed linear depth ramp: w_l = l / Σ(1..L), 1-indexed.
            # Registered as a buffer so it moves with the model and is saved
            # in state_dict, but is never updated by the optimizer.
            weights = torch.arange(1, L + 1, dtype=torch.float)
            weights = weights / weights.sum()
            self.register_buffer("depth_weights", weights)  # [L]
            self.readout = nn.Linear(projection_dim, embedding_dim)

        else:  # transformer_cls
            # Learnable CLS token prepended to the layer sequence.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, projection_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            # Positional encoding covers L+1 positions (CLS + L layers).
            self.register_buffer("pos_enc", _sinusoidal_pos_enc(L + 1, projection_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=4,
                dim_feedforward=projection_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.readout = nn.Linear(projection_dim, embedding_dim)

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

        if self.encoder_type == "weighted_sum":
            stacked = torch.stack(projected, dim=1)   # [B, L, projection_dim]
            w = self.depth_weights.view(1, -1, 1)     # [1, L, 1]
            pooled = (stacked * w).sum(dim=1)         # [B, projection_dim]
            z = self.readout(pooled)                  # [B, embedding_dim]

        else:  # transformer_cls
            B = projected[0].shape[0]
            stacked = torch.stack(projected, dim=1)           # [B, L, projection_dim]
            cls = self.cls_token.expand(B, -1, -1)            # [B, 1, projection_dim]
            tokens = torch.cat([cls, stacked], dim=1)         # [B, L+1, projection_dim]
            tokens = tokens + self.pos_enc                    # add positional encoding
            encoded = self.transformer(tokens)                # [B, L+1, projection_dim]
            z = self.readout(encoded[:, 0])                   # CLS → [B, embedding_dim]

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
