"""
Unit tests for few-shot evaluation components: EpisodeSampler, FewShotEvaluator,
EmbeddingBank, and SemanticDistanceGrouper.
Run with: pytest tests/

Mirrors the style of tests/test_losses.py: class-based, helper methods, no fixtures.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pytest

from evaluation.fewshot import EpisodeSampler, FewShotEvaluator, SemanticDistanceGrouper
from models.momentum_encoder import EmbeddingBank


# --------------------------------------------------------------------------- #
# Mock dataset helpers
# --------------------------------------------------------------------------- #

class MockDataset:
    """
    Minimal dataset that exposes class_to_indices and supports integer indexing.
    Images are random tensors [3, 32, 32]; labels are integers 0..n_classes-1.
    """
    def __init__(self, n_classes=10, n_per_class=50, img_size=32):
        self.n_classes = n_classes
        self.data = []
        self.targets = []
        self.class_to_indices = {}
        for c in range(n_classes):
            indices = []
            for i in range(n_per_class):
                idx = c * n_per_class + i
                self.data.append(torch.randn(3, img_size, img_size))
                self.targets.append(c)
                indices.append(idx)
            self.class_to_indices[c] = indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MockBackbone(torch.nn.Module):
    """Returns constant logits and a fixed-length random trajectory."""
    def __init__(self, num_classes=10, n_layers=4, hidden=64):
        super().__init__()
        self.n_layers = n_layers
        self.hidden = hidden
        self.fc = torch.nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        logits = self.fc(x.view(x.shape[0], -1))
        traj = [torch.randn(x.shape[0], self.hidden) for _ in range(self.n_layers)]
        return logits, traj


class MockMetaEncoder(torch.nn.Module):
    """Returns L2-normalised random embeddings of fixed dim."""
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, trajectory):
        B = trajectory[0].shape[0]
        z = torch.randn(B, self.embedding_dim)
        return F.normalize(z, dim=-1)


# --------------------------------------------------------------------------- #
# EpisodeSampler
# --------------------------------------------------------------------------- #

class TestEpisodeSampler:
    def _make_sampler(self, n_way=5, k_shot=1, n_query=15, n_episodes=10):
        ds = MockDataset(n_classes=10, n_per_class=50)
        return EpisodeSampler(ds, n_way=n_way, k_shot=k_shot,
                              n_query=n_query, n_episodes=n_episodes)

    def test_episode_shapes_1shot(self):
        sampler = self._make_sampler(n_way=5, k_shot=1, n_query=15, n_episodes=3)
        s_imgs, s_labels, q_imgs, q_labels = next(iter(sampler))
        assert s_imgs.shape == (5, 3, 32, 32)    # N*K = 5*1
        assert s_labels.shape == (5,)
        assert q_imgs.shape == (75, 3, 32, 32)   # N*Q = 5*15
        assert q_labels.shape == (75,)

    def test_episode_shapes_5shot(self):
        sampler = self._make_sampler(n_way=5, k_shot=5, n_query=15, n_episodes=3)
        s_imgs, s_labels, q_imgs, q_labels = next(iter(sampler))
        assert s_imgs.shape == (25, 3, 32, 32)   # N*K = 5*5
        assert q_imgs.shape == (75, 3, 32, 32)

    def test_labels_are_episode_relative(self):
        sampler = self._make_sampler(n_way=5, k_shot=1, n_query=5)
        for s_imgs, s_labels, q_imgs, q_labels in sampler:
            assert s_labels.max().item() < 5
            assert s_labels.min().item() >= 0
            assert q_labels.max().item() < 5
            assert q_labels.min().item() >= 0
            break

    def test_n_episodes_length(self):
        sampler = self._make_sampler(n_way=3, k_shot=1, n_query=5, n_episodes=20)
        count = sum(1 for _ in sampler)
        assert count == 20

    def test_len_matches_n_episodes(self):
        sampler = self._make_sampler(n_episodes=15)
        assert len(sampler) == 15

    def test_not_enough_classes_raises(self):
        ds = MockDataset(n_classes=3, n_per_class=20)
        with pytest.raises(ValueError):
            EpisodeSampler(ds, n_way=5)

    def test_all_support_labels_present(self):
        # Every episode-class (0..N-1) must appear in support labels
        sampler = self._make_sampler(n_way=5, k_shot=2, n_query=5, n_episodes=5)
        for _, s_labels, _, _ in sampler:
            assert set(s_labels.tolist()) == set(range(5))
            break


# --------------------------------------------------------------------------- #
# FewShotEvaluator
# --------------------------------------------------------------------------- #

class TestFewShotEvaluator:
    def _make_evaluator(self):
        backbone = MockBackbone()
        meta_encoder = MockMetaEncoder(embedding_dim=64)
        return FewShotEvaluator(backbone, meta_encoder, torch.device("cpu"))

    def test_accuracy_between_0_and_1(self):
        evaluator = self._make_evaluator()
        ds = MockDataset(n_classes=10, n_per_class=50)
        sampler = EpisodeSampler(ds, n_way=5, k_shot=1, n_query=5, n_episodes=10)
        results = evaluator.evaluate(sampler)
        assert 0.0 <= results["mean_acc"] <= 1.0

    def test_ci95_is_positive(self):
        evaluator = self._make_evaluator()
        ds = MockDataset(n_classes=10, n_per_class=50)
        sampler = EpisodeSampler(ds, n_way=5, k_shot=1, n_query=5, n_episodes=10)
        results = evaluator.evaluate(sampler)
        assert results["ci95"] >= 0.0

    def test_per_episode_accs_length(self):
        evaluator = self._make_evaluator()
        ds = MockDataset(n_classes=10, n_per_class=50)
        sampler = EpisodeSampler(ds, n_way=5, k_shot=1, n_query=5, n_episodes=10)
        results = evaluator.evaluate(sampler)
        assert len(results["per_episode_accs"]) == 10

    def test_prototypical_classify_perfect_separation(self):
        evaluator = self._make_evaluator()
        n_way, k_shot, D = 3, 2, 64

        # Orthogonal bases — perfect prototype separation
        bases = F.normalize(torch.eye(D)[:n_way], dim=-1)  # [3, 64]
        support_z = bases.repeat_interleave(k_shot, dim=0)  # [6, 64]
        support_labels = torch.arange(n_way).repeat_interleave(k_shot)

        # Query = exact class prototypes → should always be correct
        query_z = bases  # [3, 64]
        preds = evaluator._prototypical_classify(support_z, support_labels, query_z, n_way)
        assert (preds == torch.arange(n_way)).all()


# --------------------------------------------------------------------------- #
# EmbeddingBank
# --------------------------------------------------------------------------- #

class TestEmbeddingBank:
    def _make_bank(self, bank_size=100, embedding_dim=32):
        return EmbeddingBank(
            bank_size=bank_size,
            embedding_dim=embedding_dim,
            device=torch.device("cpu"),
        )

    def test_not_ready_before_full(self):
        bank = self._make_bank(bank_size=100)
        assert not bank.is_ready()

    def test_ready_after_full(self):
        bank = self._make_bank(bank_size=10)
        idxs = torch.arange(10)
        embs = F.normalize(torch.randn(10, 32), dim=-1)
        bank.update(idxs, embs)
        assert bank.is_ready()

    def test_partial_fill_not_ready(self):
        bank = self._make_bank(bank_size=20)
        bank.update(torch.arange(10), F.normalize(torch.randn(10, 32), dim=-1))
        assert not bank.is_ready()

    def test_update_and_retrieve_self(self):
        bank = self._make_bank(bank_size=100)
        idxs = torch.arange(10)
        embs = F.normalize(torch.randn(10, 32), dim=-1)
        bank.update(idxs, embs)
        # Query with the first embedding; nearest neighbor should be itself
        query = embs[:1]  # [1, 32]
        _, nbr_idxs = bank.get_neighbors(query, k=1, exclude_self_indices=None)
        assert nbr_idxs[0, 0].item() == 0

    def test_self_exclusion(self):
        bank = self._make_bank(bank_size=100)
        idxs = torch.arange(10)
        embs = F.normalize(torch.randn(10, 32), dim=-1)
        bank.update(idxs, embs)
        query = embs[:1]
        _, nbr_idxs = bank.get_neighbors(
            query, k=1, exclude_self_indices=torch.tensor([0])
        )
        assert nbr_idxs[0, 0].item() != 0

    def test_neighbor_embedding_shape(self):
        bank = self._make_bank(bank_size=50)
        bank.update(torch.arange(50), F.normalize(torch.randn(50, 32), dim=-1))
        query = F.normalize(torch.randn(8, 32), dim=-1)
        nbr_embs, nbr_idxs = bank.get_neighbors(query, k=3)
        assert nbr_embs.shape == (8, 3, 32)
        assert nbr_idxs.shape == (8, 3)


# --------------------------------------------------------------------------- #
# SemanticDistanceGrouper
# --------------------------------------------------------------------------- #

class TestSemanticDistanceGrouper:
    def test_close_small_mammals(self):
        grouper = SemanticDistanceGrouper()
        # Fine label 36 = hamster → superclass 16 (small_mammals) → close
        assert grouper.get_distance_group(36) == "close"

    def test_close_vehicles_1(self):
        grouper = SemanticDistanceGrouper()
        # Fine label 8 = bicycle → superclass 18 (vehicles_1) → close
        assert grouper.get_distance_group(8) == "close"

    def test_medium_reptiles(self):
        grouper = SemanticDistanceGrouper()
        # Fine label 27 = crocodile → superclass 15 (reptiles) → medium
        assert grouper.get_distance_group(27) == "medium"

    def test_distant_trees(self):
        grouper = SemanticDistanceGrouper()
        # Fine label 47 = maple_tree → superclass 17 (trees) → distant
        assert grouper.get_distance_group(47) == "distant"

    def test_group_results_aggregates_correctly(self):
        grouper = SemanticDistanceGrouper()
        # small_mammals: [36, 50, 65, 74, 80] → all close
        # trees:         [47, 52, 56, 59, 96] → all distant
        per_class_accs = {36: 0.8, 50: 0.6, 47: 0.3, 52: 0.4}
        result = grouper.group_results(per_class_accs)
        assert abs(result["close"] - 0.7) < 1e-4    # (0.8 + 0.6) / 2
        assert abs(result["distant"] - 0.35) < 1e-4 # (0.3 + 0.4) / 2

    def test_unknown_class_excluded(self):
        grouper = SemanticDistanceGrouper()
        # Fine label 0 = apple → superclass 4 (base class, not novel) → unknown
        assert grouper.get_distance_group(0) == "unknown"
