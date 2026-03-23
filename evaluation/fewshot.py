"""
Few-shot evaluation: N-way K-shot episode sampling and prototypical classification.

Three components:

EpisodeSampler
  Generates N-way K-shot episodes from any dataset that exposes a
  class_to_indices dict (same pattern as PairedCIFAR10 and CIFAR100ForTransfer).
  Each episode randomly draws N classes, samples K support + Q query images per
  class, and returns episode-relative labels (0 to N-1).

FewShotEvaluator
  Runs episodes through a frozen backbone + meta_encoder and classifies queries
  using a prototypical network in circuit embedding space: prototypes are the
  mean circuit embedding of each class's support images; queries are assigned
  to the nearest prototype by cosine distance.

  The circuit-space prototypical classifier is the natural fit for CTLS-SSL:
  the circuit scaffold hypothesis predicts that CTLS-trained embeddings produce
  better prototypes for novel classes because same-class circuit embeddings
  are more tightly clustered (higher intraclass ρ from Stage 3 results).

SemanticDistanceGrouper
  Groups CIFAR-100 fine classes by their semantic proximity to CIFAR-10 classes,
  enabling stratified analysis: the CTLS-SSL hypothesis predicts that the
  accuracy advantage over SimCLR is largest for semantically close novel classes.

  Groupings are defined by CIFAR-100 superclass identity (no external libraries):
    close:   superclass 16 (small_mammals), 18 (vehicles_1)
    medium:  superclass 15 (reptiles),      19 (vehicles_2)
    distant: superclass 17 (trees)

  Rationale:
    small_mammals and vehicles_1 are highly analogous to CIFAR-10 categories
    (cat/dog/deer and automobile/truck/airplane respectively). trees have no
    CIFAR-10 counterpart, making their circuits entirely novel.
"""

import random
from collections import defaultdict
from typing import Generator

import numpy as np
import torch
import torch.nn.functional as F

from data.ssl import _CIFAR100_FINE_TO_COARSE

# Novel superclasses and their semantic group
_NOVEL_SUPERCLASS_GROUPS = {
    15: "medium",   # reptiles   — somewhat related to frog/bird in CIFAR-10
    16: "close",    # small_mammals — analogous to cat/dog/deer
    17: "distant",  # trees      — no CIFAR-10 counterpart
    18: "close",    # vehicles_1 — analogous to automobile/truck/airplane
    19: "medium",   # vehicles_2 — partial vehicle overlap
}


# --------------------------------------------------------------------------- #
# Episode Sampler
# --------------------------------------------------------------------------- #

class EpisodeSampler:
    """
    N-way K-shot episode generator.

    Draws n_episodes episodes from a dataset with a class_to_indices attribute.
    Each episode returns (support_imgs, support_labels, query_imgs, query_labels)
    where labels are episode-relative (0 to n_way − 1), not original fine labels.
    Support and query sets are disjoint within each episode.

    Parameters
    ----------
    dataset : any dataset with class_to_indices dict[int, list[int]]
    n_way   : number of classes per episode
    k_shot  : support images per class
    n_query : query images per class (default 15, standard in few-shot literature)
    n_episodes : total episodes to generate per iteration
    """

    def __init__(
        self,
        dataset,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        n_episodes: int = 600,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes

        self.classes = list(dataset.class_to_indices.keys())
        if len(self.classes) < n_way:
            raise ValueError(
                f"Dataset has {len(self.classes)} classes but n_way={n_way}."
            )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Generator:
        for _ in range(self.n_episodes):
            yield self._sample_episode()

    def _sample_episode(self):
        episode_classes = random.sample(self.classes, self.n_way)

        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []

        for ep_label, cls in enumerate(episode_classes):
            all_idx = self.dataset.class_to_indices[cls]
            needed = self.k_shot + self.n_query
            if len(all_idx) < needed:
                # Sample with replacement if not enough samples
                chosen = random.choices(all_idx, k=needed)
            else:
                chosen = random.sample(all_idx, k=needed)

            support_pos = chosen[:self.k_shot]
            query_pos = chosen[self.k_shot:]

            for pos in support_pos:
                img, _ = self.dataset[pos]
                support_imgs.append(img)
                support_labels.append(ep_label)

            for pos in query_pos:
                img, _ = self.dataset[pos]
                query_imgs.append(img)
                query_labels.append(ep_label)

        return (
            torch.stack(support_imgs),               # [N*K, C, H, W]
            torch.tensor(support_labels, dtype=torch.long),  # [N*K]
            torch.stack(query_imgs),                 # [N*Q, C, H, W]
            torch.tensor(query_labels, dtype=torch.long),    # [N*Q]
        )


# --------------------------------------------------------------------------- #
# Few-Shot Evaluator
# --------------------------------------------------------------------------- #

class FewShotEvaluator:
    """
    Evaluates a frozen backbone + meta_encoder on N-way K-shot episodes.

    Primary classifier: prototypical network in circuit embedding space.
      Prototype = mean of support circuit embeddings per class (L2-normalised).
      Query classification = argmin cosine distance to prototype.

    The circuit-space prototypical classifier directly tests the CTLS-SSL
    hypothesis: if the circuit scaffold provides better intraclass consistency,
    same-class support embeddings should be tighter clusters, producing more
    reliable prototypes from few examples.
    """

    def __init__(self, backbone, meta_encoder, device: torch.device):
        self.backbone = backbone
        self.meta_encoder = meta_encoder
        self.device = device

    def evaluate(
        self,
        episode_sampler: EpisodeSampler,
        use_circuit_space: bool = True,
    ) -> dict:
        """
        Run all episodes and return accuracy statistics.

        Returns
        -------
        dict with keys:
          "mean_acc"        : float — mean accuracy across episodes
          "ci95"            : float — 95% confidence interval (1.96 * std / sqrt(n))
          "per_episode_accs": list[float]
        """
        self.backbone.eval()
        self.meta_encoder.eval()

        per_episode_accs = []

        for support_imgs, support_labels, query_imgs, query_labels in episode_sampler:
            support_imgs = support_imgs.to(self.device)
            query_imgs = query_imgs.to(self.device)

            support_z, support_logits = self._encode_batch(support_imgs)
            query_z, query_logits = self._encode_batch(query_imgs)

            if use_circuit_space:
                preds = self._prototypical_classify(
                    support_z, support_labels, query_z, episode_sampler.n_way
                )
            else:
                # Output-space fallback — reuse logits from above
                preds = self._prototypical_classify(
                    support_logits, support_labels, query_logits, episode_sampler.n_way
                )

            acc = (preds.cpu() == query_labels).float().mean().item()
            per_episode_accs.append(acc)

        mean_acc = float(np.mean(per_episode_accs))
        ci95 = float(1.96 * np.std(per_episode_accs) / (len(per_episode_accs) ** 0.5))

        return {
            "mean_acc": mean_acc,
            "ci95": ci95,
            "per_episode_accs": per_episode_accs,
        }

    @torch.no_grad()
    def _encode_batch(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z [N, D], logits [N, num_classes])."""
        logits, traj = self.backbone(imgs)
        z = self.meta_encoder(traj)
        return z, logits

    def _prototypical_classify(
        self,
        support_z: torch.Tensor,
        support_labels: torch.Tensor,
        query_z: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """
        Classify queries by nearest L2-normalised prototype (cosine distance).

        Prototype for class c = mean of support_z[support_labels == c],
        then L2-normalised.

        Args:
            support_z:      [N*K, D] support embeddings
            support_labels: [N*K]    episode-relative labels (0 to n_way-1)
            query_z:        [N*Q, D] query embeddings
            n_way:          number of classes in this episode

        Returns:
            preds: [N*Q] predicted episode-relative labels
        """
        D = support_z.shape[-1]
        prototypes = torch.zeros(n_way, D, device=support_z.device)

        for c in range(n_way):
            mask = support_labels.to(support_z.device) == c
            if mask.sum() > 0:
                prototypes[c] = support_z[mask].mean(dim=0)

        prototypes = F.normalize(prototypes, dim=-1)    # [n_way, D]
        query_z_norm = F.normalize(query_z, dim=-1)    # [N*Q, D]

        # Cosine similarity: [N*Q, n_way]
        sim = torch.mm(query_z_norm, prototypes.t())
        preds = sim.argmax(dim=1)  # [N*Q]
        return preds


# --------------------------------------------------------------------------- #
# Semantic Distance Grouper
# --------------------------------------------------------------------------- #

class SemanticDistanceGrouper:
    """
    Groups CIFAR-100 fine class indices by semantic distance to CIFAR-10 classes.

    Uses a hardcoded lookup based on the known CIFAR-100 superclass taxonomy.
    No external libraries required.

    Groups (for novel superclasses 15-19):
      close:   superclasses 16 (small_mammals), 18 (vehicles_1)
               fine labels: 36, 50, 65, 74, 80, 8, 13, 48, 58, 90
      medium:  superclasses 15 (reptiles), 19 (vehicles_2)
               fine labels: 27, 29, 44, 78, 93, 41, 69, 81, 85, 89
      distant: superclass 17 (trees)
               fine labels: 47, 52, 56, 59, 96
    """

    def get_distance_group(self, cifar100_fine_label: int) -> str:
        """Returns 'close', 'medium', or 'distant' for a CIFAR-100 fine label."""
        coarse = _CIFAR100_FINE_TO_COARSE[cifar100_fine_label]
        return _NOVEL_SUPERCLASS_GROUPS.get(coarse, "unknown")

    def group_results(
        self, per_class_accs: dict
    ) -> dict:
        """
        Aggregate per-class accuracies into semantic distance groups.

        Args:
            per_class_accs: dict mapping CIFAR-100 fine_label → accuracy

        Returns:
            dict with keys 'close', 'medium', 'distant' → mean accuracy
        """
        groups: dict[str, list[float]] = defaultdict(list)
        for fine_label, acc in per_class_accs.items():
            group = self.get_distance_group(fine_label)
            if group != "unknown":
                groups[group].append(acc)

        return {
            g: float(np.mean(accs)) if accs else float("nan")
            for g, accs in groups.items()
        }
