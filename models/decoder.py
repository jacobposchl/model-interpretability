"""
CircuitDecoder D: circuit embedding z → reconstructed input x̂.

Inverts the meta-encoder/backbone mapping: given the 64-dim L2-normalised
circuit embedding z produced by the CTLS meta-encoder, reconstruct an
approximation of the original input image.

Architecture: an MLP stem expands z to a small spatial feature map, then
three transposed-convolution blocks upsample to CIFAR-10's 32×32 output.

    z [B, 64]
    → Linear(64, 256·4·4) + GELU  → reshape [B, 256, 4, 4]
    → ConvTranspose(256→128) + BN + GELU  → [B, 128, 8, 8]
    → ConvTranspose(128→64)  + BN + GELU  → [B, 64, 16, 16]
    → ConvTranspose(64→32)   + BN + GELU  → [B, 32, 32, 32]
    → Conv2d(32→3, 3×3)                   → [B, 3, 32, 32]

Training mode: post-hoc. The backbone and meta-encoder are frozen; only
decoder weights are updated. This tests whether the already-structured
CTLS circuit space contains enough visual information to support generation.

Output is in normalised pixel space (same mean/std as the CIFAR-10
training data). Denormalise before display.
"""

import torch
import torch.nn as nn


class CircuitDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_channels: list[int] = None,
        input_spatial: int = 4,
        output_channels: int = 3,
    ):
        """
        Args:
            embedding_dim:   dimensionality of the input circuit embedding z
            hidden_channels: channel counts at each ConvTranspose stage
                             [c0, c1, c2, c3]; initial spatial map has c0 channels,
                             and the final Conv2d maps c3 → output_channels.
                             Defaults to [256, 128, 64, 32].
            input_spatial:   spatial size of the initial feature map before
                             upsampling (4 gives 4→8→16→32 via three stages).
            output_channels: number of output image channels (3 for RGB).
        """
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 128, 64, 32]

        c0, c1, c2, c3 = hidden_channels
        self.input_spatial = input_spatial

        # Project z to initial spatial feature map
        self.stem = nn.Sequential(
            nn.Linear(embedding_dim, c0 * input_spatial * input_spatial),
            nn.GELU(),
        )

        # Three upsampling stages: 4 → 8 → 16 → 32
        self.up1 = _up_block(c0, c1)  # [B, c0, 4, 4]  → [B, c1, 8, 8]
        self.up2 = _up_block(c1, c2)  # [B, c1, 8, 8]  → [B, c2, 16, 16]
        self.up3 = _up_block(c2, c3)  # [B, c2, 16, 16] → [B, c3, 32, 32]

        # Final projection to pixel space (no activation — output is in normalised space)
        self.out_conv = nn.Conv2d(c3, output_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: circuit embedding, shape [B, embedding_dim] (L2-normalised)

        Returns:
            x_hat: reconstructed image, shape [B, 3, 32, 32],
                   in normalised pixel space (same as CIFAR-10 training data).
                   Denormalise before display.
        """
        B = z.shape[0]
        h = self.stem(z)
        h = h.view(B, -1, self.input_spatial, self.input_spatial)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return self.out_conv(h)


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Transposed-conv upsample ×2 with BatchNorm + GELU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )
