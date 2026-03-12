"""
Unit tests for loss functions.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from losses.consistency import CircuitConsistencyLoss, _depth_weights
from losses.contrastive import NTXentLoss


# --------------------------------------------------------------------------- #
# depth_weights
# --------------------------------------------------------------------------- #

class TestDepthWeights:
    def test_linear_sums_to_one(self):
        w = _depth_weights(8, "linear", torch.device("cpu"))
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_uniform_sums_to_one(self):
        w = _depth_weights(8, "uniform", torch.device("cpu"))
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_exponential_sums_to_one(self):
        w = _depth_weights(8, "exponential", torch.device("cpu"))
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_linear_is_increasing(self):
        w = _depth_weights(8, "linear", torch.device("cpu"))
        for i in range(len(w) - 1):
            assert w[i] <= w[i + 1]

    def test_uniform_all_equal(self):
        w = _depth_weights(5, "uniform", torch.device("cpu"))
        assert torch.allclose(w, torch.full((5,), 1.0 / 5))

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError):
            _depth_weights(4, "unknown", torch.device("cpu"))


# --------------------------------------------------------------------------- #
# CircuitConsistencyLoss
# --------------------------------------------------------------------------- #

class TestCircuitConsistencyLoss:
    def _make_traj(self, B=4, dims=None):
        dims = dims or [64, 64, 128, 256]
        return [torch.randn(B, d) for d in dims]

    def test_identical_trajectories_zero_loss(self):
        loss_fn = CircuitConsistencyLoss(weight_scheme="linear", distance="l2")
        traj = self._make_traj()
        loss = loss_fn(traj, traj)
        assert loss.item() < 1e-5

    def test_different_trajectories_positive_loss(self):
        loss_fn = CircuitConsistencyLoss()
        t1 = self._make_traj()
        t2 = self._make_traj()
        loss = loss_fn(t1, t2)
        assert loss.item() > 0

    def test_loss_is_scalar(self):
        loss_fn = CircuitConsistencyLoss()
        t1, t2 = self._make_traj(), self._make_traj()
        loss = loss_fn(t1, t2)
        assert loss.shape == torch.Size([])

    def test_cosine_distance(self):
        loss_fn = CircuitConsistencyLoss(distance="cosine")
        t1, t2 = self._make_traj(), self._make_traj()
        loss = loss_fn(t1, t2)
        assert loss.item() >= 0

    def test_mismatched_lengths_raises(self):
        loss_fn = CircuitConsistencyLoss()
        t1 = self._make_traj(dims=[64, 128])
        t2 = self._make_traj(dims=[64, 128, 256])
        with pytest.raises(AssertionError):
            loss_fn(t1, t2)

    def test_uniform_gives_equal_layer_weights(self):
        # With uniform weighting, reversing layer order should give same loss
        loss_fn = CircuitConsistencyLoss(weight_scheme="uniform")
        t1 = self._make_traj()
        t2 = self._make_traj()
        loss_fwd = loss_fn(t1, t2)
        loss_rev = loss_fn(list(reversed(t1)), list(reversed(t2)))
        assert abs(loss_fwd.item() - loss_rev.item()) < 1e-4

    def test_backprop_through_loss(self):
        loss_fn = CircuitConsistencyLoss()
        t1 = [torch.randn(4, 64, requires_grad=True) for _ in range(4)]
        t2 = [torch.randn(4, 64) for _ in range(4)]
        loss = loss_fn(t1, t2)
        loss.backward()
        assert all(t.grad is not None for t in t1)


# --------------------------------------------------------------------------- #
# NTXentLoss
# --------------------------------------------------------------------------- #

class TestNTXentLoss:
    def test_output_is_scalar(self):
        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        loss = loss_fn(z1, z2)
        assert loss.shape == torch.Size([])

    def test_positive_loss(self):
        loss_fn = NTXentLoss()
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        assert loss_fn(z1, z2).item() > 0

    def test_identical_pairs_lower_than_random(self):
        # When z1==z2 (perfect alignment), loss should be lower than for random pairs
        loss_fn = NTXentLoss(temperature=0.07)
        z = torch.randn(16, 64)
        loss_identical = loss_fn(z, z.clone())
        loss_random = loss_fn(z, torch.randn(16, 64))
        assert loss_identical.item() < loss_random.item()

    def test_backprop(self):
        loss_fn = NTXentLoss()
        z1 = torch.randn(8, 64, requires_grad=True)
        z2 = torch.randn(8, 64, requires_grad=True)
        loss = loss_fn(z1, z2)
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None
