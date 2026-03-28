"""
Circuit space visualization for Phase 1 validation.

Provides per-layer UMAP plots, alignment profile heatmaps, circuit member
image grids, span coverage diagrams, and multi-circuit membership histograms.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from evaluation.circuit_analysis import CIFAR10_CLASSES, denormalize

_COLORS = cm.get_cmap("tab10", 10)
_COLOR_LIST = [_COLORS(i) for i in range(10)]


def plot_per_layer_umap(
    z_list: list[np.ndarray],
    labels: np.ndarray,
    layer_indices: list[int] | None = None,
    max_samples: int = 3000,
) -> plt.Figure:
    """
    UMAP of z_l colored by class label, one subplot per layer.

    Args:
        z_list:        list of L arrays, each [N, d]
        labels:        [N] class labels
        layer_indices: which layers to plot (default: all)
        max_samples:   subsample for speed

    Returns:
        matplotlib Figure
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    if layer_indices is None:
        layer_indices = list(range(len(z_list)))

    n_layers = len(layer_indices)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    # Subsample
    N = z_list[0].shape[0]
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
    else:
        idx = np.arange(N)

    for i, l in enumerate(layer_indices):
        z_sub = z_list[l][idx]
        labels_sub = labels[idx]

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)
        z_2d = reducer.fit_transform(z_sub)

        ax = axes[i]
        for cls in range(10):
            mask = labels_sub == cls
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                c=[_COLOR_LIST[cls]], label=CIFAR10_CLASSES[cls],
                s=3, alpha=0.5, rasterized=True,
            )
        ax.set_title(f"Layer {l+1}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    _add_legend(fig)
    fig.suptitle("Per-Layer Circuit Space (UMAP)", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_profile_heatmap(
    profiles: np.ndarray,
    labels: np.ndarray | None = None,
    max_pairs: int = 500,
) -> plt.Figure:
    """
    Heatmap of alignment profiles [N_pairs, L].

    Args:
        profiles:  [N_pairs, L] alignment profile vectors
        labels:    optional [N_pairs] for sorting by class
        max_pairs: max pairs to display

    Returns:
        matplotlib Figure
    """
    if profiles.shape[0] > max_pairs:
        idx = np.random.choice(profiles.shape[0], max_pairs, replace=False)
        profiles = profiles[idx]
        if labels is not None:
            labels = labels[idx]

    if labels is not None:
        sort_idx = np.argsort(labels)
        profiles = profiles[sort_idx]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(profiles, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pair index")
    ax.set_title("Alignment Profiles")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    return fig


def plot_circuit_members(
    images: np.ndarray,
    labels: np.ndarray,
    profiles: np.ndarray,
    span: tuple[int, int],
    n_show: int = 16,
) -> plt.Figure:
    """
    Display sample images from a circuit cluster with profile annotations.

    Args:
        images:   [N, 3, 32, 32] images (denormalized, [0,1])
        labels:   [N] class labels
        profiles: [N, L] per-input average profile (mean over pair partners)
        span:     (l_start, l_end) circuit span
        n_show:   how many to display

    Returns:
        matplotlib Figure
    """
    n_show = min(n_show, len(images))
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i in range(n_show):
        ax = axes[i]
        img = images[i].transpose(1, 2, 0)  # [H, W, 3]
        ax.imshow(np.clip(img, 0, 1))
        cls_name = CIFAR10_CLASSES[labels[i]] if labels[i] < 10 else str(labels[i])
        span_vals = profiles[i, span[0]:span[1]+1]
        span_str = ", ".join(f"{v:.2f}" for v in span_vals)
        ax.set_title(f"{cls_name}\nspan [{span[0]+1}-{span[1]+1}]: {span_str}",
                     fontsize=8)
        ax.axis("off")

    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Circuit Members — Span [{span[0]+1}, {span[1]+1}]",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def plot_span_coverage(
    circuits: list[dict],
    n_layers: int,
) -> plt.Figure:
    """
    Visualize which spans have canonical circuits discovered.

    Args:
        circuits: list of circuit dicts with 'span' and 'size' keys
        n_layers: total number of backbone layers

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(circuits) * 0.4)))

    for i, circuit in enumerate(circuits):
        l_start, l_end = circuit["span"]
        size = circuit["size"]
        ax.barh(i, l_end - l_start + 1, left=l_start, height=0.6,
                color=_COLOR_LIST[i % 10], alpha=0.7,
                edgecolor="black", linewidth=0.5)
        ax.text(l_end + 0.3, i, f"n={size}", va="center", fontsize=8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Circuit #")
    ax.set_xlim(-0.5, n_layers + 1)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{l+1}" for l in range(n_layers)])
    ax.set_title("Canonical Circuit Span Coverage")
    fig.tight_layout()
    return fig


def plot_multi_circuit_histogram(
    membership_counts: np.ndarray,
) -> plt.Figure:
    """
    Distribution of per-pair circuit membership counts.

    Args:
        membership_counts: [N_pairs] number of circuits each pair belongs to

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    max_count = int(membership_counts.max())
    bins = np.arange(0, max_count + 2) - 0.5
    ax.hist(membership_counts, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of canonical circuits")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Multi-Circuit Membership Distribution")
    ax.set_xticks(range(max_count + 1))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _add_legend(fig):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_COLOR_LIST[i], label=CIFAR10_CLASSES[i],
                   markersize=8)
        for i in range(10)
    ]
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=9)
