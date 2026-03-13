"""
Decoder visualisation utilities for Stage 6 (Circuit Decoder).

Four analyses:
  1. Reconstruction grid  — original image | CTLS recon | baseline recon
  2. Centroid decoding    — decode the mean circuit embedding per class →
                            the model's "prototype" image for each category
  3. Interpolation grid   — linearly interpolate between two class centroids
                            and decode each step; reveals the visual ambiguity zone
  4. Confusion zone       — find real images from class A whose circuit
                            embedding is closest to class B's centroid; decode them
                            to visualise which visual features cause confusion

Usage:
    from evaluation.decoder_viz import DecoderVisualizer

    viz = DecoderVisualizer(backbone, meta_encoder, decoder_ctls, val_loader, device)
    fig = viz.centroid_decoding()
    fig.savefig('experiments/decoder/centroids_ctls.png', dpi=150, bbox_inches='tight')
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 normalisation constants
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
_STD  = torch.tensor([0.2470, 0.2435, 0.2616])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Convert normalised CIFAR-10 pixel tensor → [0, 1] for display.

    Accepts [C, H, W] or [B, C, H, W].
    """
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    if x.dim() == 4:
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    else:
        mean = mean[:, None, None]
        std  = std[:, None, None]
    return (x * std + mean).clamp(0, 1)


class DecoderVisualizer:
    def __init__(
        self,
        backbone,
        meta_encoder,
        decoder,
        loader: DataLoader,
        device: torch.device,
    ):
        self.backbone = backbone
        self.meta_encoder = meta_encoder
        self.decoder = decoder
        self.loader = loader
        self.device = device

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def collect_all(self, max_samples: int = 5000):
        """Collect (z, x_original, label) from the loader."""
        self.backbone.eval()
        self.meta_encoder.eval()
        self.decoder.eval()

        all_z, all_x, all_labels = [], [], []
        n = 0

        for batch in self.loader:
            # Handle both paired (x, x2, label) and standard (x, label) loaders
            x = batch[0].to(self.device)
            labels = batch[-1]

            _, traj = self.backbone(x)
            z = self.meta_encoder(traj)

            all_z.append(z.cpu())
            all_x.append(x.cpu())
            all_labels.append(labels.cpu())

            n += x.shape[0]
            if n >= max_samples:
                break

        z = torch.cat(all_z, dim=0)[:max_samples]
        x = torch.cat(all_x, dim=0)[:max_samples]
        labels = torch.cat(all_labels, dim=0)[:max_samples]
        return z, x, labels

    def _class_centroid(self, z: torch.Tensor, labels: torch.Tensor, cls: int) -> torch.Tensor:
        """Return the L2-normalised mean embedding for a class."""
        mask = labels == cls
        c = z[mask].mean(dim=0)
        return F.normalize(c, dim=-1)

    # ------------------------------------------------------------------ #
    # 1. Reconstruction grid
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def reconstruction_grid(
        self,
        n_images: int = 8,
        decoder_baseline=None,
        title: str = "Circuit Decoder Reconstructions",
    ) -> plt.Figure:
        """
        Show a grid of: [original | CTLS reconstruction | baseline reconstruction].

        decoder_baseline: optional second decoder trained on baseline embeddings.
                          If provided, a third row is added for direct comparison.
        """
        self.decoder.eval()

        # Take the first batch from the loader
        x_batch, labels_batch = None, None
        for batch in self.loader:
            x_batch = batch[0].to(self.device)[:n_images]
            labels_batch = batch[-1][:n_images]
            break

        _, traj = self.backbone(x_batch)
        z = self.meta_encoder(traj)
        x_hat_ctls = self.decoder(z)

        rows_data = [x_batch, x_hat_ctls]
        row_labels = ["Original", "CTLS recon"]

        if decoder_baseline is not None:
            decoder_baseline.eval()
            x_hat_base = decoder_baseline(z)
            rows_data.append(x_hat_base)
            row_labels.append("Baseline recon")

        n_rows = len(rows_data)
        fig, axes = plt.subplots(n_rows, n_images, figsize=(n_images * 1.6, n_rows * 1.8))
        fig.suptitle(title, fontsize=12, y=1.01)

        for row_idx, (row_data, row_label) in enumerate(zip(rows_data, row_labels)):
            for col_idx in range(n_images):
                ax = axes[row_idx, col_idx]
                img = denormalize(row_data[col_idx]).permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=9)
                if row_idx == 0:
                    ax.set_title(CIFAR10_CLASSES[labels_batch[col_idx].item()], fontsize=8)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 2. Centroid decoding — "Universal X"
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def centroid_decoding(
        self,
        max_samples: int = 5000,
        title: str = "Class Centroids — The Model's Prototype for Each Category",
    ) -> plt.Figure:
        """
        Compute the mean circuit embedding per class and decode each centroid.
        The resulting images show what the model's circuits consider the
        canonical visual form of each class.
        """
        self.decoder.eval()
        z, _, labels = self.collect_all(max_samples)

        centroids = torch.stack(
            [self._class_centroid(z, labels, cls) for cls in range(10)],
            dim=0,
        ).to(self.device)  # [10, D]

        decoded = self.decoder(centroids)  # [10, 3, 32, 32]

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(title, fontsize=12)
        axes = axes.flatten()

        for cls in range(10):
            img = denormalize(decoded[cls]).permute(1, 2, 0).cpu().numpy()
            axes[cls].imshow(img)
            axes[cls].set_title(CIFAR10_CLASSES[cls], fontsize=10)
            axes[cls].axis("off")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 3. Interpolation grid
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def interpolation_grid(
        self,
        class_a: int,
        class_b: int,
        n_steps: int = 8,
        max_samples: int = 5000,
        title: str = None,
    ) -> plt.Figure:
        """
        Linearly interpolate between the centroid of class_a and class_b,
        re-normalise at each step, then decode.

        The resulting images reveal the visual ambiguity zone between two classes —
        the shared features that make the model treat them as related.

        Args:
            class_a, class_b: integer class indices (0–9)
            n_steps:          number of interpolation steps (including endpoints)
        """
        self.decoder.eval()
        z, _, labels = self.collect_all(max_samples)

        z_a = self._class_centroid(z, labels, class_a)
        z_b = self._class_centroid(z, labels, class_b)

        alphas = torch.linspace(0, 1, n_steps)
        interp_z = torch.stack(
            [F.normalize((1 - a) * z_a + a * z_b, dim=-1) for a in alphas],
            dim=0,
        ).to(self.device)  # [n_steps, D]

        decoded = self.decoder(interp_z)  # [n_steps, 3, 32, 32]

        name_a = CIFAR10_CLASSES[class_a]
        name_b = CIFAR10_CLASSES[class_b]
        if title is None:
            title = f"Circuit space interpolation: {name_a} → {name_b}"

        fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.6, 2.2))
        fig.suptitle(title, fontsize=11)

        for i in range(n_steps):
            img = denormalize(decoded[i]).permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(img)
            axes[i].axis("off")
            if i == 0:
                axes[i].set_title(name_a, fontsize=8)
            elif i == n_steps - 1:
                axes[i].set_title(name_b, fontsize=8)
            else:
                axes[i].set_title(f"α={alphas[i].item():.2f}", fontsize=7)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 4. Confusion zone
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def confusion_zone(
        self,
        class_a: int,
        class_b: int,
        n_images: int = 8,
        max_samples: int = 5000,
        title: str = None,
    ) -> plt.Figure:
        """
        Find real images from class_a whose circuit embedding is closest
        (by cosine distance) to the centroid of class_b. Decode them.

        These are the images where the model's internal reasoning most
        resembles its reasoning for class_b — the "confused" examples.
        The decoded images highlight which visual features drive the confusion.

        Args:
            class_a: source class (we look at images from this class)
            class_b: target class (we find images closest to this centroid)
        """
        self.decoder.eval()
        z, x_orig, labels = self.collect_all(max_samples)

        centroid_b = self._class_centroid(z, labels, class_b)

        mask_a = labels == class_a
        z_a = z[mask_a]
        x_a = x_orig[mask_a]

        # Cosine distance: 1 − z · centroid_b  (lower = closer to class_b)
        dists = 1 - (z_a @ centroid_b)
        closest_idx = dists.argsort()[:n_images]

        x_confused = x_a[closest_idx].to(self.device)
        _, traj = self.backbone(x_confused)
        z_confused = self.meta_encoder(traj)
        x_hat = self.decoder(z_confused)

        name_a = CIFAR10_CLASSES[class_a]
        name_b = CIFAR10_CLASSES[class_b]
        if title is None:
            title = (
                f"Confusion zone: {name_a} images "
                f"whose circuits most resemble {name_b}"
            )

        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.6, 3.5))
        fig.suptitle(title, fontsize=10)

        for col in range(n_images):
            orig  = denormalize(x_confused[col]).permute(1, 2, 0).cpu().numpy()
            recon = denormalize(x_hat[col]).permute(1, 2, 0).cpu().numpy()
            axes[0, col].imshow(orig)
            axes[0, col].axis("off")
            axes[1, col].imshow(recon)
            axes[1, col].axis("off")
            if col == 0:
                axes[0, col].set_ylabel("Original", fontsize=8)
                axes[1, col].set_ylabel("Decoded z", fontsize=8)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # 5. Quantitative MSE comparison
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def mse_by_class(self, max_samples: int = 5000) -> dict[str, list[float]]:
        """
        Compute per-class reconstruction MSE.

        Returns:
            {'airplane': mse, 'automobile': mse, ...}
        """
        self.decoder.eval()
        z, x_orig, labels = self.collect_all(max_samples)

        results = {}
        for cls in range(10):
            mask = labels == cls
            z_cls  = z[mask].to(self.device)
            x_cls  = x_orig[mask].to(self.device)
            x_hat  = self.decoder(z_cls)
            mse    = F.mse_loss(x_hat, x_cls).item()
            results[CIFAR10_CLASSES[cls]] = mse

        return results
