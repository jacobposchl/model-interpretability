"""
Stage 5 evaluation: SAE-based monosemanticity scoring.

A Sparse Autoencoder (SAE) is trained on the activations at each layer.
After training, each SAE feature (dictionary element) can be analysed for
how selectively it fires for a single class vs. multiple classes.

A monosemantic feature fires strongly and consistently for one category.
A polysemantic feature fires for many categories — a sign of superposition.

Metrics:
  monosemanticity_score — fraction of features where one class accounts for
                          >80% of total activation (configurable threshold).
  feature_purity        — for each feature, entropy of the per-class
                          activation distribution. Low entropy = monosemantic.
  circuit_reuse         — fraction of features that are active (above a
                          threshold) for >1 category. Measures how much the
                          model shares circuits across semantically distinct
                          inputs.

This connects CTLS to the existing mechanistic interpretability literature
and provides a quantitative bridge to SAE-based work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Sparse Autoencoder
# --------------------------------------------------------------------------- #

class SparseAutoencoder(nn.Module):
    """
    1-hidden-layer overcomplete autoencoder with L1 sparsity.

    Encoder: input_dim → dict_size  (ReLU activation = non-negative features)
    Decoder: dict_size → input_dim  (tied or untied weights)
    """

    def __init__(self, input_dim: int, dict_size: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.l1_coeff = l1_coeff
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

        # Normalise decoder columns to unit norm (standard SAE init)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.encoder(x))          # [B, dict_size]
        recon = self.decoder(features)               # [B, input_dim]
        return recon, features

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, features = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        sparsity_loss = self.l1_coeff * features.abs().mean()
        return recon_loss + sparsity_loss


# --------------------------------------------------------------------------- #
# Trainer for SAE
# --------------------------------------------------------------------------- #

class SAETrainer:
    def __init__(self, sae: SparseAutoencoder, device: torch.device):
        self.sae = sae
        self.device = device
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    def train_on_activations(self, activations: torch.Tensor, epochs: int = 50):
        """
        activations: [N, input_dim] tensor of collected layer activations.
        """
        self.sae.train()
        dataset = torch.utils.data.TensorDataset(activations)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for (x,) in loader:
                x = x.to(self.device)
                loss = self.sae.loss(x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Renormalize decoder columns after each step
                with torch.no_grad():
                    self.sae.decoder.weight.data = F.normalize(
                        self.sae.decoder.weight.data, dim=0
                    )
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  SAE epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.5f}")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

class MonosemanticityScorer:
    """
    Trains SAEs on each layer of the backbone and computes monosemanticity
    metrics comparing a CTLS model against a baseline.
    """

    def __init__(
        self,
        backbone,
        loader: DataLoader,
        device: torch.device,
        dict_size_multiplier: int = 4,
        l1_coeff: float = 1e-3,
        mono_threshold: float = 0.8,
        active_threshold: float = 0.01,
    ):
        self.backbone = backbone
        self.loader = loader
        self.device = device
        self.dict_size_multiplier = dict_size_multiplier
        self.l1_coeff = l1_coeff
        self.mono_threshold = mono_threshold
        self.active_threshold = active_threshold

    @torch.no_grad()
    def _collect_layer_activations(self, layer_idx: int, n_samples: int = 10000):
        """Collect activations at a specific trajectory layer index."""
        self.backbone.eval()
        all_acts, all_labels = [], []
        n = 0

        for batch in self.loader:
            if len(batch) == 3:
                x, _, labels = batch
            else:
                x, labels = batch
            x = x.to(self.device)
            _, traj = self.backbone(x)
            all_acts.append(traj[layer_idx].cpu())
            all_labels.append(labels)
            n += x.shape[0]
            if n >= n_samples:
                break

        acts = torch.cat(all_acts, dim=0)[:n_samples]
        labels = torch.cat(all_labels, dim=0)[:n_samples]
        return acts, labels

    def _compute_metrics(
        self,
        sae: SparseAutoencoder,
        activations: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int = 10,
    ) -> dict:
        """Compute monosemanticity metrics for a trained SAE."""
        sae.eval()
        with torch.no_grad():
            _, features = sae(activations.to(self.device))
            features = features.cpu().numpy()  # [N, dict_size]

        labels_np = labels.numpy()
        dict_size = features.shape[1]

        # Per-class mean activation for each feature: [num_classes, dict_size]
        class_means = np.zeros((num_classes, dict_size))
        for cls in range(num_classes):
            mask = labels_np == cls
            if mask.sum() > 0:
                class_means[cls] = features[mask].mean(axis=0)

        total_activation = class_means.sum(axis=0) + 1e-8  # [dict_size]
        class_fractions = class_means / total_activation    # [num_classes, dict_size]
        max_fraction = class_fractions.max(axis=0)          # [dict_size]

        # Monosemanticity score: fraction of features dominated by one class
        mono_score = float((max_fraction > self.mono_threshold).mean())

        # Feature purity: entropy of class activation distribution per feature
        eps = 1e-10
        entropy = -(class_fractions * np.log(class_fractions + eps)).sum(axis=0)
        mean_entropy = float(entropy.mean())

        # Circuit reuse: fraction of features active (above threshold) for
        # more than one class
        active_per_class = (class_means > self.active_threshold)   # [num_classes, dict_size]
        n_active_classes = active_per_class.sum(axis=0)            # [dict_size]
        reuse_rate = float((n_active_classes > 1).mean())

        return {
            "monosemanticity_score": mono_score,
            "mean_feature_entropy": mean_entropy,
            "circuit_reuse_rate": reuse_rate,
            "max_class_fraction_mean": float(max_fraction.mean()),
        }

    def score_all_layers(
        self,
        sae_epochs: int = 50,
        n_samples: int = 10000,
    ) -> list[dict]:
        """
        Train one SAE per layer and return metrics for each.
        """
        n_layers = len(self.backbone.layer_dims)
        results = []

        for layer_idx in range(n_layers):
            print(f"Layer {layer_idx+1}/{n_layers}: collecting activations...")
            acts, labels = self._collect_layer_activations(layer_idx, n_samples)

            input_dim = acts.shape[-1]
            dict_size = input_dim * self.dict_size_multiplier
            sae = SparseAutoencoder(input_dim, dict_size, self.l1_coeff).to(self.device)

            print(f"  Training SAE (input_dim={input_dim}, dict_size={dict_size})...")
            trainer = SAETrainer(sae, self.device)
            trainer.train_on_activations(acts, epochs=sae_epochs)

            metrics = self._compute_metrics(sae, acts, labels)
            metrics["layer_idx"] = layer_idx
            results.append(metrics)

            print(
                f"  mono={metrics['monosemanticity_score']:.3f} "
                f"entropy={metrics['mean_feature_entropy']:.3f} "
                f"reuse={metrics['circuit_reuse_rate']:.3f}"
            )

        return results

    def compare_with_baseline(
        self,
        baseline_backbone,
        sae_epochs: int = 50,
        n_samples: int = 10000,
    ) -> dict:
        """
        Run monosemanticity scoring on both CTLS and baseline backbone.
        Returns a comparison dict with delta metrics.
        """
        print("Scoring CTLS model...")
        ctls_results = self.score_all_layers(sae_epochs, n_samples)

        original_backbone = self.backbone
        self.backbone = baseline_backbone
        print("Scoring baseline model...")
        base_results = self.score_all_layers(sae_epochs, n_samples)
        self.backbone = original_backbone

        comparison = []
        for ctls, base in zip(ctls_results, base_results):
            comparison.append({
                "layer_idx": ctls["layer_idx"],
                "ctls_mono": ctls["monosemanticity_score"],
                "base_mono": base["monosemanticity_score"],
                "delta_mono": ctls["monosemanticity_score"] - base["monosemanticity_score"],
                "ctls_reuse": ctls["circuit_reuse_rate"],
                "base_reuse": base["circuit_reuse_rate"],
                "delta_reuse": ctls["circuit_reuse_rate"] - base["circuit_reuse_rate"],
            })

        return {"layer_results": comparison, "ctls": ctls_results, "baseline": base_results}
