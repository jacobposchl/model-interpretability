"""
Stage 3 evaluation: verify that the circuit latent space contains information
that the output embedding space discards.

Core experiment: find pairs of inputs that are (a) same-category, (b) produce
similar output embeddings (model "agrees" on the classification), but (c) have
visually different surface properties (e.g. one is a clear image, one is
perturbed / occluded). Then measure distances in both spaces.

If the circuit space carries genuinely new information, it should show
larger distances for these pairs than the output space does.

Two comparison modes:
  1. Synthetic degradation  — apply Gaussian noise or random occlusion to
     create a "degraded" version of each image. The degraded and clean images
     are same-category and (if the model is robust) similar in output space,
     but take different computational paths internally.
  2. Natural within-class pairs — sort val-set pairs by their output-space
     distance vs their circuit-space distance and report the rank-order
     correlation. High correlation = circuit space ≈ output space (bad).
     Low / inverted correlation for a subset = circuit space captures more.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class EmbeddingComparator:
    def __init__(self, backbone, meta_encoder, device: torch.device):
        self.backbone = backbone
        self.meta_encoder = meta_encoder
        self.device = device

    @torch.no_grad()
    def compare_clean_vs_degraded(
        self,
        loader: DataLoader,
        noise_std: float = 0.3,
        n_samples: int = 512,
    ) -> dict:
        """
        For each image x in the loader, create a degraded version x_deg by
        adding Gaussian noise. Measure:
          - output_dist:  cosine distance in softmax space
          - circuit_dist: cosine distance in circuit embedding space

        Returns summary statistics and per-pair arrays.
        """
        self.backbone.eval()
        self.meta_encoder.eval()

        out_dists, circ_dists, labels_list = [], [], []
        n = 0

        for batch in loader:
            if len(batch) == 3:
                x, _, labels = batch
            else:
                x, labels = batch
            x = x.to(self.device)

            # Degraded version: add Gaussian noise, clamp to valid range
            x_deg = x + noise_std * torch.randn_like(x)
            x_deg = x_deg.clamp(-3.0, 3.0)  # within normalised range

            # Forward pass: both in one batch
            B = x.shape[0]
            x_cat = torch.cat([x, x_deg], dim=0)
            logits_cat, traj_cat = self.backbone(x_cat)
            z_cat = self.meta_encoder(traj_cat)

            logits_clean = F.softmax(logits_cat[:B], dim=-1)
            logits_deg   = F.softmax(logits_cat[B:], dim=-1)
            z_clean = z_cat[:B]
            z_deg   = z_cat[B:]

            # Pairwise distances (one per image)
            out_d = 1.0 - F.cosine_similarity(logits_clean, logits_deg, dim=-1)
            circ_d = 1.0 - F.cosine_similarity(z_clean, z_deg, dim=-1)

            out_dists.append(out_d.cpu())
            circ_dists.append(circ_d.cpu())
            labels_list.append(labels)

            n += B
            if n >= n_samples:
                break

        out_dists  = torch.cat(out_dists)[:n_samples].numpy()
        circ_dists = torch.cat(circ_dists)[:n_samples].numpy()

        return {
            "output_dist_mean":  float(out_dists.mean()),
            "circuit_dist_mean": float(circ_dists.mean()),
            "ratio_circuit_over_output": float(circ_dists.mean() / (out_dists.mean() + 1e-8)),
            "output_dists":  out_dists,
            "circuit_dists": circ_dists,
        }

    def plot_distance_comparison(
        self,
        loader: DataLoader,
        noise_std: float = 0.3,
        n_samples: int = 512,
        title: str = "Circuit vs Output Space: clean–degraded pair distances",
    ) -> plt.Figure:
        results = self.compare_clean_vs_degraded(loader, noise_std, n_samples)
        out_d   = results["output_dists"]
        circ_d  = results["circuit_dists"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=13)

        # Histograms
        axes[0].hist(out_d, bins=40, alpha=0.7, color="steelblue", label="output space")
        axes[0].hist(circ_d, bins=40, alpha=0.7, color="darkorange", label="circuit space")
        axes[0].set_xlabel("Cosine distance (clean vs degraded)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distance distributions")
        axes[0].legend()

        # Scatter
        axes[1].scatter(out_d, circ_d, s=4, alpha=0.4, color="purple")
        axes[1].set_xlabel("Output space distance")
        axes[1].set_ylabel("Circuit space distance")
        axes[1].set_title("Per-pair: output vs circuit distance")
        lim = max(out_d.max(), circ_d.max()) * 1.05
        axes[1].plot([0, lim], [0, lim], "k--", lw=0.8, label="y=x")
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        return fig

    @torch.no_grad()
    def intraclass_distance_rank(
        self,
        loader: DataLoader,
        n_samples: int = 1000,
    ) -> dict:
        """
        Collect circuit and output embeddings for n_samples items. For each
        class, compute all pairwise distances in both spaces and return the
        Spearman rank correlation between them.

        Low correlation for a class = circuit space provides genuinely
        different ordering than output space (good: new information).
        """
        from scipy.stats import spearmanr

        self.backbone.eval()
        self.meta_encoder.eval()

        all_z, all_logits, all_labels = [], [], []
        n = 0
        for batch in loader:
            if len(batch) == 3:
                x, _, labels = batch
            else:
                x, labels = batch
            x = x.to(self.device)
            logits, traj = self.backbone(x)
            z = self.meta_encoder(traj)
            all_z.append(z.cpu())
            all_logits.append(F.softmax(logits, dim=-1).cpu())
            all_labels.append(labels)
            n += x.shape[0]
            if n >= n_samples:
                break

        z = torch.cat(all_z, dim=0)[:n_samples].numpy()
        logits = torch.cat(all_logits, dim=0)[:n_samples].numpy()
        labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()

        from sklearn.metrics.pairwise import cosine_distances
        results = {}
        for cls in range(10):
            mask = labels == cls
            if mask.sum() < 5:
                continue
            dz = cosine_distances(z[mask]).flatten()
            dl = cosine_distances(logits[mask]).flatten()
            rho, pval = spearmanr(dz, dl)
            results[cls] = {"spearman_rho": float(rho), "p_value": float(pval)}

        return results
