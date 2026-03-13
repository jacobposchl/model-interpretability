"""
Unit tests for loss functions.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from losses.consistency import CircuitConsistencyLoss, _depth_weights
from losses.contrastive import SupConLoss


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
        loss_fn = CircuitConsistencyLoss(weight_scheme="linear")
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

    def test_loss_bounded(self):
        # Cosine distance is in [0, 1], so weighted sum is in [0, 1]
        loss_fn = CircuitConsistencyLoss()
        t1, t2 = self._make_traj(), self._make_traj()
        loss = loss_fn(t1, t2)
        assert 0.0 <= loss.item() <= 1.0

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
# SupConLoss
# --------------------------------------------------------------------------- #

class TestSupConLoss:
    def test_output_is_scalar(self):
        loss_fn = SupConLoss(temperature=0.07)
        z = torch.randn(16, 64)
        labels = torch.randint(0, 10, (16,))
        loss = loss_fn(z, labels)
        assert loss.shape == torch.Size([])

    def test_positive_loss(self):
        loss_fn = SupConLoss()
        z = torch.randn(16, 64)
        labels = torch.randint(0, 10, (16,))
        assert loss_fn(z, labels).item() > 0

    def test_same_class_lower_loss_than_all_different(self):
        # Embeddings that are already aligned by class should have lower loss
        # than completely random embeddings with scattered labels
        loss_fn = SupConLoss(temperature=0.07)
        # Clustered: each class has tight embeddings
        z_clustered = torch.zeros(20, 64)
        labels = torch.arange(10).repeat(2)  # 10 classes × 2 samples
        for c in range(10):
            mask = labels == c
            z_clustered[mask] = torch.randn(64) + c * 5.0  # well-separated
        loss_clustered = loss_fn(z_clustered, labels)
        loss_random = loss_fn(torch.randn(20, 64), labels)
        assert loss_clustered.item() < loss_random.item()

    def test_backprop(self):
        loss_fn = SupConLoss()
        z = torch.randn(16, 64, requires_grad=True)
        labels = torch.randint(0, 10, (16,))
        loss = loss_fn(z, labels)
        loss.backward()
        assert z.grad is not None
