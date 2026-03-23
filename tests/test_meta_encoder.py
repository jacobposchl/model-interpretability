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
        # At low τ, σ(a/τ) → 1 for a>0 and 0 for a<0
        assert out[0].item() > 0
        assert abs(out[1].item()) < 0.01

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
# Shared helpers
# --------------------------------------------------------------------------- #

LAYER_DIMS = [64, 64, 128, 128, 256, 256, 512, 512]  # ResNet18


def make_traj(B=4, layer_dims=LAYER_DIMS):
    return [torch.randn(B, d) for d in layer_dims]


# --------------------------------------------------------------------------- #
# MetaEncoder — weighted_sum
# --------------------------------------------------------------------------- #

class TestMetaEncoderWeightedSum:
    def test_output_shape(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="weighted_sum")
        z = enc(make_traj())
        assert z.shape == (4, 64)

    def test_output_is_unit_norm(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="weighted_sum")
        z = enc(make_traj())
        assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)

    def test_depth_weights_sum_to_one(self):
        enc = MetaEncoder(LAYER_DIMS, encoder_type="weighted_sum")
        assert abs(enc.depth_weights.sum().item() - 1.0) < 1e-5

    def test_depth_weights_are_increasing(self):
        enc = MetaEncoder(LAYER_DIMS, encoder_type="weighted_sum")
        w = enc.depth_weights
        assert (w[1:] >= w[:-1]).all()

    def test_different_batch_sizes(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="weighted_sum")
        for B in [1, 8, 32]:
            z = enc(make_traj(B=B))
            assert z.shape == (B, 64)

    def test_backprop(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="weighted_sum")
        traj = [torch.randn(4, d, requires_grad=True) for d in LAYER_DIMS]
        z = enc(traj)
        z.sum().backward()
        assert all(t.grad is not None for t in traj)

    def test_wrong_num_layers_raises(self):
        enc = MetaEncoder(LAYER_DIMS, encoder_type="weighted_sum")
        with pytest.raises(AssertionError):
            enc([torch.randn(4, 64) for _ in range(3)])


# --------------------------------------------------------------------------- #
# MetaEncoder — transformer_cls
# --------------------------------------------------------------------------- #

class TestMetaEncoderTransformerCLS:
    def test_output_shape(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="transformer_cls", projection_dim=128)
        z = enc(make_traj())
        assert z.shape == (4, 64)

    def test_output_is_unit_norm(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="transformer_cls", projection_dim=128)
        z = enc(make_traj())
        assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)

    def test_different_batch_sizes(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="transformer_cls", projection_dim=128)
        for B in [1, 8, 16]:
            z = enc(make_traj(B=B))
            assert z.shape == (B, 64)

    def test_backprop(self):
        enc = MetaEncoder(LAYER_DIMS, embedding_dim=64, encoder_type="transformer_cls", projection_dim=128)
        traj = [torch.randn(4, d, requires_grad=True) for d in LAYER_DIMS]
        z = enc(traj)
        z.sum().backward()
        assert all(t.grad is not None for t in traj)

    def test_projection_dim_must_be_divisible_by_4(self):
        # projection_dim=130 is not divisible by 4 — transformer nhead=4 will fail
        enc = MetaEncoder(LAYER_DIMS, encoder_type="transformer_cls", projection_dim=130)
        with pytest.raises(Exception):
            enc(make_traj())

    def test_wrong_num_layers_raises(self):
        enc = MetaEncoder(LAYER_DIMS, encoder_type="transformer_cls", projection_dim=128)
        with pytest.raises(AssertionError):
            enc([torch.randn(4, 64) for _ in range(3)])


# --------------------------------------------------------------------------- #
# MetaEncoder — shared
# --------------------------------------------------------------------------- #

class TestMetaEncoderShared:
    def test_unknown_encoder_type_raises(self):
        with pytest.raises(ValueError):
            MetaEncoder(LAYER_DIMS, encoder_type="rnn")

    def test_both_variants_produce_same_output_dim(self):
        traj = make_traj()
        enc_a = MetaEncoder(LAYER_DIMS, embedding_dim=32, encoder_type="weighted_sum")
        enc_b = MetaEncoder(LAYER_DIMS, embedding_dim=32, encoder_type="transformer_cls", projection_dim=128)
        assert enc_a(traj).shape == enc_b(traj).shape
