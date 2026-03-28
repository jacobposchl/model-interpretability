"""
Unit tests for span-centric circuit discovery.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from evaluation.discovery import SpanCentricDiscovery


# --------------------------------------------------------------------------- #
# Span Enumeration
# --------------------------------------------------------------------------- #

class TestSpanEnumeration:
    def test_count_for_8_layers(self):
        disc = SpanCentricDiscovery(n_layers=8)
        spans = disc.enumerate_spans()
        # L(L+1)/2 = 8*9/2 = 36
        assert len(spans) == 36

    def test_count_for_4_layers(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        assert len(spans) == 10  # 4*5/2

    def test_spans_are_valid(self):
        disc = SpanCentricDiscovery(n_layers=8)
        spans = disc.enumerate_spans()
        for l_start, l_end in spans:
            assert 0 <= l_start <= l_end < 8

    def test_single_layer_spans_included(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        for l in range(4):
            assert (l, l) in spans

    def test_full_range_span_included(self):
        disc = SpanCentricDiscovery(n_layers=4)
        spans = disc.enumerate_spans()
        assert (0, 3) in spans


# --------------------------------------------------------------------------- #
# Within-Span Sharpening
# --------------------------------------------------------------------------- #

class TestWithinSpanSharpening:
    def test_output_sums_to_one(self):
        disc = SpanCentricDiscovery(n_layers=8, tau_discovery=0.5)
        subvec = np.random.rand(10, 5)
        sharpened = disc.sharpen_within_span(subvec)
        row_sums = sharpened.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_low_temperature_concentrates(self):
        disc = SpanCentricDiscovery(n_layers=8, tau_discovery=0.01)
        subvec = np.array([[0.1, 0.5, 0.9, 0.2, 0.3]])
        sharpened = disc.sharpen_within_span(subvec)
        # Should concentrate on index 2 (highest value)
        assert sharpened[0, 2] > 0.9

    def test_high_temperature_is_uniform(self):
        disc = SpanCentricDiscovery(n_layers=8, tau_discovery=100.0)
        subvec = np.array([[0.1, 0.5, 0.9, 0.2, 0.3]])
        sharpened = disc.sharpen_within_span(subvec)
        # Should be nearly uniform
        expected = np.ones(5) / 5
        np.testing.assert_allclose(sharpened[0], expected, atol=0.05)

    def test_shape_preserved(self):
        disc = SpanCentricDiscovery(n_layers=8)
        subvec = np.random.rand(20, 3)
        sharpened = disc.sharpen_within_span(subvec)
        assert sharpened.shape == (20, 3)


# --------------------------------------------------------------------------- #
# Extract Span Sub-Vector
# --------------------------------------------------------------------------- #

class TestExtractSpanSubvector:
    def test_correct_slice(self):
        profiles = np.random.rand(10, 8)  # 10 pairs, 8 layers
        subvec = SpanCentricDiscovery.extract_span_subvector(profiles, (2, 5))
        assert subvec.shape == (10, 4)  # layers 2, 3, 4, 5
        np.testing.assert_array_equal(subvec, profiles[:, 2:6])

    def test_single_layer_span(self):
        profiles = np.random.rand(5, 8)
        subvec = SpanCentricDiscovery.extract_span_subvector(profiles, (3, 3))
        assert subvec.shape == (5, 1)

    def test_full_span(self):
        profiles = np.random.rand(5, 8)
        subvec = SpanCentricDiscovery.extract_span_subvector(profiles, (0, 7))
        assert subvec.shape == (5, 8)
        np.testing.assert_array_equal(subvec, profiles)


# --------------------------------------------------------------------------- #
# Canonicality Filter
# --------------------------------------------------------------------------- #

class TestCanonicalityFilter:
    def test_filters_small_clusters(self):
        disc = SpanCentricDiscovery(n_layers=8, min_cluster_fraction=0.1)
        # 100 pairs: cluster 0 has 50, cluster 1 has 5, noise has 45
        labels = np.array([0]*50 + [1]*5 + [-1]*45)
        canonical = disc.filter_canonical(labels, n_total_pairs=100)
        assert 0 in canonical
        assert 1 not in canonical  # 5/100 = 0.05 < 0.1

    def test_noise_excluded(self):
        disc = SpanCentricDiscovery(n_layers=8)
        labels = np.array([-1]*100)
        canonical = disc.filter_canonical(labels, n_total_pairs=100)
        assert len(canonical) == 0


# --------------------------------------------------------------------------- #
# Multi-Circuit Membership
# --------------------------------------------------------------------------- #

class TestMultiCircuitMembership:
    def test_no_circuits_gives_zeros(self):
        counts = SpanCentricDiscovery.multi_circuit_membership([], n_pairs=10)
        assert counts.shape == (10,)
        assert counts.sum() == 0

    def test_counts_accumulate(self):
        n = 20
        c1 = {"pair_mask": np.array([True]*10 + [False]*10)}
        c2 = {"pair_mask": np.array([False]*5 + [True]*10 + [False]*5)}
        counts = SpanCentricDiscovery.multi_circuit_membership([c1, c2], n_pairs=n)
        # Indices 5-9 are in both circuits
        assert counts[7] == 2
        # Index 0 is in only circuit 1
        assert counts[0] == 1
        # Index 15 is in neither
        assert counts[15] == 0
