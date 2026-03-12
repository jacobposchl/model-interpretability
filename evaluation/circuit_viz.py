"""
Circuit latent space visualisation.

Generates UMAP and t-SNE projections of circuit embeddings and compares them
against the model's output (logit) embeddings. The key visual claim to
validate: CTLS training should produce tighter, more semantically coherent
clusters in circuit space than a baseline model, and the circuit space should
show structure that the output embedding space does not capture.

Usage:
    from evaluation.circuit_viz import CircuitVisualizer
    viz = CircuitVisualizer(backbone, meta_encoder, val_loader, device)
    fig = viz.plot_umap(title="Stage 2 — CTLS circuit space")
    fig.savefig("experiments/ctls/umap_circuit.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class CircuitVisualizer:
    def __init__(self, backbone, meta_encoder, loader: DataLoader, device: torch.device):
        self.backbone = backbone
        self.meta_encoder = meta_encoder
        self.loader = loader
        self.device = device

    @torch.no_grad()
    def collect_embeddings(self, max_samples: int = 5000):
        """
        Run the val loader and collect:
          - circuit embeddings z  [N, embedding_dim]
          - output (logit) vectors [N, num_classes]
          - labels                 [N]
        """
        self.backbone.eval()
        self.meta_encoder.eval()

        all_z, all_logits, all_labels = [], [], []
        n = 0

        for batch in self.loader:
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
            if n >= max_samples:
                break

        z = torch.cat(all_z, dim=0)[:max_samples].numpy()
        logits = torch.cat(all_logits, dim=0)[:max_samples].numpy()
        labels = torch.cat(all_labels, dim=0)[:max_samples].numpy()
        return z, logits, labels

    def plot_umap(
        self,
        title: str = "Circuit Latent Space (UMAP)",
        max_samples: int = 5000,
        compare_output: bool = True,
    ) -> plt.Figure:
        """
        Plot UMAP of circuit embeddings. If compare_output=True, also plots
        UMAP of output (softmax) embeddings side-by-side for direct comparison.
        """
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        z, logits, labels = self.collect_embeddings(max_samples)
        colors = cm.get_cmap("tab10", 10)
        color_list = [colors(i) for i in range(10)]

        n_plots = 2 if compare_output else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
        if n_plots == 1:
            axes = [axes]

        # Circuit space UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
        z_2d = reducer.fit_transform(z)
        _scatter_2d(axes[0], z_2d, labels, color_list, f"{title}\n(circuit space)")

        if compare_output:
            reducer2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
            logits_2d = reducer2.fit_transform(logits)
            _scatter_2d(
                axes[1], logits_2d, labels, color_list,
                f"{title}\n(output / softmax space)"
            )

        _add_legend(fig, color_list)
        fig.tight_layout()
        return fig

    def plot_tsne(
        self,
        title: str = "Circuit Latent Space (t-SNE)",
        max_samples: int = 2000,
        compare_output: bool = True,
    ) -> plt.Figure:
        from sklearn.manifold import TSNE

        z, logits, labels = self.collect_embeddings(max_samples)
        colors = cm.get_cmap("tab10", 10)
        color_list = [colors(i) for i in range(10)]

        n_plots = 2 if compare_output else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
        if n_plots == 1:
            axes = [axes]

        z_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)
        _scatter_2d(axes[0], z_2d, labels, color_list, f"{title}\n(circuit space)")

        if compare_output:
            logits_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(logits)
            _scatter_2d(
                axes[1], logits_2d, labels, color_list,
                f"{title}\n(output / softmax space)"
            )

        _add_legend(fig, color_list)
        fig.tight_layout()
        return fig

    @torch.no_grad()
    def cluster_separation_score(self, max_samples: int = 5000) -> dict:
        """
        Compute silhouette score for circuit space vs output space.
        Higher silhouette = tighter, more separated clusters.
        Used to quantitatively compare Stage 1 vs Stage 2 results.
        """
        from sklearn.metrics import silhouette_score

        z, logits, labels = self.collect_embeddings(max_samples)

        # Subsample for speed if needed
        if len(labels) > 2000:
            idx = np.random.choice(len(labels), 2000, replace=False)
            z, logits, labels = z[idx], logits[idx], labels[idx]

        sil_circuit = silhouette_score(z, labels, metric="cosine")
        sil_output = silhouette_score(logits, labels, metric="cosine")

        return {
            "silhouette_circuit": float(sil_circuit),
            "silhouette_output": float(sil_output),
            "delta": float(sil_circuit - sil_output),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _scatter_2d(ax, xy, labels, color_list, title):
    for cls in range(10):
        mask = labels == cls
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=[color_list[cls]], label=CIFAR10_CLASSES[cls],
            s=5, alpha=0.6, rasterized=True,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def _add_legend(fig, color_list):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_list[i], label=CIFAR10_CLASSES[i], markersize=8)
        for i in range(10)
    ]
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=9)
