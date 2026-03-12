"""
Magnitude-weighted soft masking gate.

S_i = σ(a_i / τ) · a_i

where a_i is the pre-activation value and τ is the temperature parameter.

At high τ: smooth, continuous weighting — activations are gently scaled.
As τ → 0: approaches a binary hard mask — neurons are either on or off.

Temperature is annealed externally by the training scheduler via set_temperature().
Using pre-activation values is important: neurons close to (but below) threshold
carry information that post-ReLU representations discard entirely.
"""

import torch
import torch.nn as nn


class SoftMask(nn.Module):
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = init_temperature

    def set_temperature(self, tau: float):
        self.temperature = max(tau, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(x / self.temperature)
        return gate * x
