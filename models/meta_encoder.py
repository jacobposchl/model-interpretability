"""
Meta-encoder: trajectory -> per-layer circuit representations z_1, ..., z_L.

Takes the full activation trajectory T(x) = (h_1, h_2, ..., h_L) — a list of
L2-normalized, per-layer activation vectors — and produces per-layer circuit
representations via a transformer with Rotary Position Embeddings (RoPE).

Key design choices:
  - Each layer is projected to a common projection_dim d first, since layers
    have different widths. This makes the encoder backbone-agnostic.
  - Projector: Linear -> GELU -> LayerNorm (per-layer, learned independently).
  - RoPE encodes layer depth by rotating Q/K vectors, so the dot product
    between layer tokens naturally decays with their relative depth distance.
    This gives the transformer an inductive bias favoring contiguous circuits.
  - No CLS token. Input [p_1, ..., p_L] -> transformer -> [z_1, ..., z_L].
    Each z_l is L2-normalized.
  - ProfileRegressor: MLP that takes z_l^a * z_l^b (element-wise product)
    and predicts the per-layer cosine similarity s_l(a, b) from the backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary Position Embeddings (RoPE)
# --------------------------------------------------------------------------- #

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding applied to Q and K vectors.

    For each position index, rotates pairs of dimensions by an angle
    proportional to the position, so that the dot product between two
    position-encoded vectors depends on their relative distance.
    """

    def __init__(self, dim: int, max_positions: int = 64):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        # Frequency bands: theta_i = 1 / 10000^(2i/d)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)  # [dim/2]

        # Precompute cos/sin for all positions up to max_positions
        positions = torch.arange(max_positions).float()  # [max_pos]
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [max_pos, dim/2]
        self.register_buffer("cos_cached", angles.cos())  # [max_pos, dim/2]
        self.register_buffer("sin_cached", angles.sin())  # [max_pos, dim/2]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to Q and K.

        Args:
            q, k: [B, n_heads, seq_len, head_dim]
            seq_len: sequence length (number of layer tokens)

        Returns:
            Rotated (q, k) with same shape.
        """
        cos = self.cos_cached[:seq_len]  # [seq_len, head_dim/2]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting: [1, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = _apply_rotary(q, cos, sin)
        k_rot = _apply_rotary(k, cos, sin)
        return q_rot, k_rot


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embedding to x: [B, n_heads, seq_len, head_dim].
    Splits head_dim into pairs and rotates each pair.
    """
    # Split into even and odd dimensions
    x1 = x[..., 0::2]  # [B, n_heads, seq_len, head_dim/2]
    x2 = x[..., 1::2]
    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # Interleave back
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# --------------------------------------------------------------------------- #
# Custom Multi-Head Attention with RoPE
# --------------------------------------------------------------------------- #

class RoPEMultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model]
        """
        B, S, _ = x.shape

        # Project to Q, K, V and reshape to [B, n_heads, S, head_dim]
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, seq_len=S)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, n_heads, S, S]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v  # [B, n_heads, S, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_output)


# --------------------------------------------------------------------------- #
# Transformer Encoder Layer with RoPE (pre-norm)
# --------------------------------------------------------------------------- #

class RoPETransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer with RoPE attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Meta-Encoder
# --------------------------------------------------------------------------- #

class MetaEncoder(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        projection_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            layer_dims:           List of activation dimensions per backbone layer.
            projection_dim:       Common dimension d for per-layer projections.
                                  Must be divisible by n_heads.
            n_heads:              Number of attention heads in the transformer.
            n_transformer_layers: Number of transformer encoder layers.
            dropout:              Dropout rate.
        """
        super().__init__()
        assert projection_dim % n_heads == 0, (
            f"projection_dim ({projection_dim}) must be divisible by n_heads ({n_heads})"
        )

        self.layer_dims = layer_dims
        self.projection_dim = projection_dim
        L = len(layer_dims)

        # Per-layer projectors: Linear -> GELU -> LayerNorm
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, projection_dim),
                nn.GELU(),
                nn.LayerNorm(projection_dim),
            )
            for d in layer_dims
        ])

        # RoPE transformer encoder
        self.transformer_layers = nn.ModuleList([
            RoPETransformerLayer(
                d_model=projection_dim,
                n_heads=n_heads,
                dim_feedforward=projection_dim * 2,
                dropout=dropout,
            )
            for _ in range(n_transformer_layers)
        ])

        # Final layer norm (post-transformer)
        self.final_norm = nn.LayerNorm(projection_dim)

    def forward(self, trajectory: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            trajectory: list of L tensors, each shape [B, D_l] (L2-normalized)

        Returns:
            list of L tensors, each shape [B, projection_dim], L2-normalised
        """
        assert len(trajectory) == len(self.projectors), (
            f"Trajectory has {len(trajectory)} layers but encoder expects "
            f"{len(self.projectors)}"
        )

        # Project each layer to common dimension
        projected = [proj(h) for proj, h in zip(self.projectors, trajectory)]

        # Stack into sequence: [B, L, d]
        x = torch.stack(projected, dim=1)

        # Pass through RoPE transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.final_norm(x)

        # Split back into per-layer tensors, L2-normalize each
        z_list = []
        for l in range(x.shape[1]):
            z_l = F.normalize(x[:, l], dim=-1)
            z_list.append(z_l)

        return z_list


# --------------------------------------------------------------------------- #
# Profile Regressor
# --------------------------------------------------------------------------- #

class ProfileRegressor(nn.Module):
    """
    MLP that takes the element-wise product z_l^a * z_l^b and predicts the
    per-layer backbone cosine similarity s_l(a, b).

    The element-wise product is symmetric (swapping a and b gives the same
    vector), which is correct since cosine similarity is symmetric.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_product: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_product: [N, d] — element-wise product of two z vectors

        Returns:
            [N] — predicted similarity (unbounded, MSE loss handles range)
        """
        return self.mlp(z_product).squeeze(-1)
