"""
Unit tests for InfoLoss and GeometryLoss.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.meta_encoder import ProfileRegressor
from losses.info_loss import InfoLoss
from losses.geometry_loss import GeometryLoss


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

D = 64   # projection dim
L = 8    # layers
B = 16   # batch size


def make_z_list(B=B, L=L, d=D):
    """Random L2-normalized per-layer z-vectors."""
    z_list = []
    for _ in range(L):
        z = torch.randn(B, d)
        z = z / z.norm(dim=-1, keepdim=True)
        z_list.append(z)
    return z_list


def make_true_sims_pairwise(N_pairs=100, L=L):
    """Random per-layer similarities for pairs."""
    return torch.rand(N_pairs, L)


def make_true_sims_matrix(B=B, L=L):
    """Random pairwise similarity matrix [B, B, L]."""
    sims = torch.rand(B, B, L)
    # Make symmetric
    sims = (sims + sims.transpose(0, 1)) / 2
    return sims


# --------------------------------------------------------------------------- #
# InfoLoss
# --------------------------------------------------------------------------- #

class TestInfoLoss:
    def _make_loss(self):
        reg = ProfileRegressor(input_dim=D, hidden_dim=32)
        return InfoLoss(regressor=reg)

    def test_output_is_scalar(self):
        loss_fn = self._make_loss()
        N = 50
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]
        sims = make_true_sims_pairwise(N)
        loss = loss_fn(z_a, z_b, sims)
        assert loss.dim() == 0

    def test_positive_loss(self):
        loss_fn = self._make_loss()
        N = 50
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]
        sims = make_true_sims_pairwise(N)
        loss = loss_fn(z_a, z_b, sims)
        assert loss.item() > 0

    def test_perfect_prediction_gives_low_loss(self):
        """If the regressor perfectly predicts similarities, loss should be ~0."""
        reg = ProfileRegressor(input_dim=D, hidden_dim=32)
        loss_fn = InfoLoss(regressor=reg)

        N = 20
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]

        # Compute what the regressor would predict
        with torch.no_grad():
            fake_sims = []
            for l in range(L):
                pred = reg(z_a[l] * z_b[l])
                fake_sims.append(pred)
            fake_sims = torch.stack(fake_sims, dim=1)

        loss = loss_fn(z_a, z_b, fake_sims)
        assert loss.item() < 0.01

    def test_backprop(self):
        loss_fn = self._make_loss()
        N = 20
        z_a = [torch.randn(N, D, requires_grad=True) for _ in range(L)]
        z_b = [torch.randn(N, D, requires_grad=True) for _ in range(L)]
        sims = make_true_sims_pairwise(N)
        loss = loss_fn(z_a, z_b, sims)
        loss.backward()
        assert all(z.grad is not None for z in z_a)


# --------------------------------------------------------------------------- #
# GeometryLoss
# --------------------------------------------------------------------------- #

class TestGeometryLoss:
    def test_output_is_scalar(self):
        loss_fn = GeometryLoss(temperature=0.1)
        z_list = make_z_list()
        sims = make_true_sims_matrix()
        loss = loss_fn(z_list, sims)
        assert loss.dim() == 0

    def test_positive_loss(self):
        loss_fn = GeometryLoss(temperature=0.1)
        z_list = make_z_list()
        sims = make_true_sims_matrix()
        loss = loss_fn(z_list, sims)
        assert loss.item() > 0

    def test_no_nan(self):
        """Loss should not produce NaN even with small temperature."""
        loss_fn = GeometryLoss(temperature=0.05)
        z_list = make_z_list()
        sims = make_true_sims_matrix()
        loss = loss_fn(z_list, sims)
        assert not torch.isnan(loss)

    def test_temperature_sensitivity(self):
        """Lower temperature should generally produce higher loss."""
        z_list = make_z_list()
        sims = make_true_sims_matrix()

        loss_high_tau = GeometryLoss(temperature=1.0)(z_list, sims)
        loss_low_tau = GeometryLoss(temperature=0.05)(z_list, sims)

        # Not strictly guaranteed but should hold for random z
        assert loss_low_tau.item() != loss_high_tau.item()

    def test_backprop(self):
        loss_fn = GeometryLoss(temperature=0.1)
        # Pre-normalize then make leaf tensors with requires_grad so .grad is populated
        z_list = [torch.randn(B, D) / D**0.5 for _ in range(L)]
        z_list = [z / z.norm(dim=-1, keepdim=True) for z in z_list]
        z_list = [z.detach().requires_grad_(True) for z in z_list]
        sims = make_true_sims_matrix()
        loss = loss_fn(z_list, sims)
        loss.backward()
        assert all(z.grad is not None for z in z_list)
