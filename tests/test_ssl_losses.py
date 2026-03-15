"""
Unit tests for SSL loss functions: NTXentLoss and DINOLoss.
Run with: pytest tests/

Mirrors the style of tests/test_losses.py exactly:
  - Class-based test organisation
  - No pytest fixtures — helper methods on each class
  - Assertions use abs(x - y) < epsilon for floats
  - Gradient tests check .grad is not None after backward()
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pytest

from losses.simclr import NTXentLoss
from losses.dino_loss import DINOLoss, DINOProjectionHead


# --------------------------------------------------------------------------- #
# NTXentLoss
# --------------------------------------------------------------------------- #

class TestNTXentLoss:
    def _make_views(self, N=16, D=64):
        z1 = F.normalize(torch.randn(N, D), dim=-1)
        z2 = F.normalize(torch.randn(N, D), dim=-1)
        return z1, z2

    def test_output_is_scalar(self):
        loss_fn = NTXentLoss(temperature=0.1)
        z1, z2 = self._make_views()
        assert loss_fn(z1, z2).shape == torch.Size([])

    def test_positive_loss(self):
        loss_fn = NTXentLoss(temperature=0.1)
        z1, z2 = self._make_views()
        assert loss_fn(z1, z2).item() > 0

    def test_identical_pairs_lower_loss_than_random(self):
        # z2 = z1 (perfect augmentation invariance) should give lower loss
        loss_fn = NTXentLoss(temperature=0.1)
        z1, _ = self._make_views()
        loss_perfect = loss_fn(z1, z1.clone())
        loss_random = loss_fn(z1, F.normalize(torch.randn(16, 64), dim=-1))
        assert loss_perfect.item() < loss_random.item()

    def test_backprop(self):
        loss_fn = NTXentLoss(temperature=0.1)
        z1 = torch.randn(8, 64, requires_grad=True)
        z2 = torch.randn(8, 64)
        loss_fn(
            F.normalize(z1, dim=-1),
            F.normalize(z2, dim=-1),
        ).backward()
        assert z1.grad is not None

    def test_symmetry(self):
        # NTXentLoss(z1, z2) should equal NTXentLoss(z2, z1)
        loss_fn = NTXentLoss(temperature=0.1)
        z1, z2 = self._make_views(N=8)
        forward = loss_fn(z1, z2).item()
        backward = loss_fn(z2, z1).item()
        assert abs(forward - backward) < 1e-4

    def test_higher_temperature_lower_loss(self):
        # At very high temperature, all similarities are ~equal → lower contrastive loss
        z1, z2 = self._make_views()
        loss_low = NTXentLoss(temperature=0.1)(z1, z2)
        loss_high = NTXentLoss(temperature=10.0)(z1, z2)
        assert loss_low.item() > loss_high.item()

    def test_single_sample_raises(self):
        # With N=1, there are no negatives — loss is ill-defined (log(0))
        # Not a crash requirement but good to document; just ensure no exception
        # is raised (the clamp + 1e-8 should handle it gracefully)
        loss_fn = NTXentLoss(temperature=0.1)
        z1 = F.normalize(torch.randn(1, 64), dim=-1)
        z2 = F.normalize(torch.randn(1, 64), dim=-1)
        loss = loss_fn(z1, z2)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# DINOLoss
# --------------------------------------------------------------------------- #

class TestDINOLoss:
    def test_output_is_scalar(self):
        loss_fn = DINOLoss(out_dim=64, teacher_temp=0.04, student_temp=0.1)
        s = torch.randn(16, 64)
        t = torch.randn(16, 64)
        assert loss_fn(s, t).shape == torch.Size([])

    def test_positive_loss(self):
        loss_fn = DINOLoss(out_dim=64)
        s, t = torch.randn(16, 64), torch.randn(16, 64)
        assert loss_fn(s, t).item() > 0

    def test_center_updates_after_forward(self):
        loss_fn = DINOLoss(out_dim=64, center_momentum=0.9)
        initial_center = loss_fn.center.clone()
        s, t = torch.randn(16, 64), torch.randn(16, 64)
        loss_fn(s, t)
        assert not torch.allclose(loss_fn.center, initial_center)

    def test_center_has_no_grad_fn(self):
        # Center is updated in-place using detached teacher output;
        # the buffer itself must not carry a gradient computation graph.
        loss_fn = DINOLoss(out_dim=64)
        loss_fn(torch.randn(8, 64), torch.randn(8, 64))
        assert loss_fn.center.grad_fn is None

    def test_backprop_through_student(self):
        loss_fn = DINOLoss(out_dim=64)
        s = torch.randn(8, 64, requires_grad=True)
        t = torch.randn(8, 64)
        loss_fn(s, t).backward()
        assert s.grad is not None

    def test_repeated_forward_accumulates_center(self):
        # Running multiple forward passes should shift center away from zero
        loss_fn = DINOLoss(out_dim=64, center_momentum=0.0)  # instant update
        t = torch.ones(8, 64) * 5.0  # strong non-zero teacher output
        loss_fn(torch.randn(8, 64), t)
        # With momentum=0, center = teacher_output.mean(0) immediately
        expected = t.mean(dim=0, keepdim=True)
        assert torch.allclose(loss_fn.center, expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# DINOProjectionHead
# --------------------------------------------------------------------------- #

class TestDINOProjectionHead:
    def test_output_shape(self):
        head = DINOProjectionHead(in_dim=64, hidden_dim=128, out_dim=256, bottleneck_dim=32)
        z = torch.randn(8, 64)
        out = head(z)
        assert out.shape == (8, 256)

    def test_backprop(self):
        head = DINOProjectionHead(in_dim=64, out_dim=128)
        z = torch.randn(8, 64, requires_grad=True)
        head(z).sum().backward()
        assert z.grad is not None

    def test_output_not_normalised(self):
        # The head outputs raw logits (normalisation happens inside DINOLoss)
        head = DINOProjectionHead(in_dim=64, out_dim=64)
        z = torch.randn(4, 64)
        out = head(z)
        norms = out.norm(dim=-1)
        # Norms should NOT all be 1.0 (the last linear has no L2 normalisation)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.1)
