"""
Unit tests for MetaEncoder and SoftMask.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.soft_mask import SoftMask
from models.meta_encoder import MetaEncoder


# --------------------------------------------------------------------------- #
# SoftMask
# --------------------------------------------------------------------------- #

class TestSoftMask:
    def test_output_shape_preserved(self):
        mask = SoftMask(init_temperature=1.0)
        x = torch.randn(8, 64)
        assert mask(x).shape == x.shape

    def test_high_temperature_is_smooth(self):
        mask = SoftMask(init_temperature=100.0)
        x = torch.tensor([1.0, -1.0, 0.0])
        out = mask(x)
        # At very high τ, σ(a/τ) ≈ 0.5, so out ≈ 0.5 * a
        assert torch.allclose(out, 0.5 * x, atol=0.01)

    def test_low_temperature_approaches_relu(self):
        mask = SoftMask(init_temperature=0.001)
        x = torch.tensor([2.0, -2.0, 0.5])
        out = mask(x)
        relu = torch.relu(x)
        # At low τ, σ(a/τ) → 1 for a>0 and 0 for a<0
        # so out ≈ relu(a) for large |a|
        assert out[0].item() > 0     # positive input
        assert abs(out[1].item()) < 0.01  # negative input → near zero

    def test_set_temperature(self):
        mask = SoftMask(init_temperature=1.0)
        mask.set_temperature(0.5)
        assert mask.temperature == 0.5

    def test_temperature_floor(self):
        mask = SoftMask()
        mask.set_temperature(0.0)
        assert mask.temperature >= 1e-6

    def test_gradients_flow_through(self):
        mask = SoftMask(init_temperature=1.0)
        x = torch.randn(4, 32, requires_grad=True)
        out = mask(x)
        out.sum().backward()
        assert x.grad is not None


# --------------------------------------------------------------------------- #
# MetaEncoder (MLP variant)
# --------------------------------------------------------------------------- #

class TestMetaEncoderMLP:
    LAYER_DIMS = [64, 64, 128, 128, 256, 256, 512, 512]

    def _make_traj(self, B=4):
        return [torch.randn(B, d) for d in self.LAYER_DIMS]

    def test_output_shape(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="mlp")
        z = enc(self._make_traj())
        assert z.shape == (4, 64)

    def test_output_is_unit_norm(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="mlp")
        z = enc(self._make_traj())
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_different_batch_sizes(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="mlp")
        for B in [1, 8, 32]:
            traj = [torch.randn(B, d) for d in self.LAYER_DIMS]
            z = enc(traj)
            assert z.shape == (B, 64)

    def test_backprop(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="mlp")
        traj = [torch.randn(4, d, requires_grad=True) for d in self.LAYER_DIMS]
        z = enc(traj)
        z.sum().backward()
        assert all(t.grad is not None for t in traj)

    def test_wrong_num_layers_raises(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="mlp")
        short_traj = [torch.randn(4, 64) for _ in range(3)]  # wrong length
        with pytest.raises(AssertionError):
            enc(short_traj)


# --------------------------------------------------------------------------- #
# MetaEncoder (Transformer variant)
# --------------------------------------------------------------------------- #

class TestMetaEncoderTransformer:
    LAYER_DIMS = [64, 128, 256, 512]

    def _make_traj(self, B=4):
        return [torch.randn(B, d) for d in self.LAYER_DIMS]

    def test_output_shape(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="transformer")
        z = enc(self._make_traj())
        assert z.shape == (4, 64)

    def test_output_is_unit_norm(self):
        enc = MetaEncoder(self.LAYER_DIMS, embedding_dim=64, encoder_type="transformer")
        z = enc(self._make_traj())
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_unknown_encoder_type_raises(self):
        with pytest.raises(ValueError):
            MetaEncoder(self.LAYER_DIMS, encoder_type="rnn")
