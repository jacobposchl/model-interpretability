"""
Circuit analysis utilities for Stage 6 — Circuit Analysis.

Two experiments:
  1. Activation Maximization — optimise input pixels to minimise cosine
     distance to a target point in circuit space ("circuit dreams").
  2. KNN Retrieval — k-nearest neighbours in circuit space vs output space,
     revealing where the model's internal representations and final predictions
     diverge.

Usage:
    from evaluation.circuit_analysis import CircuitAnalyzer, denormalize, CIFAR10_CLASSES
    analyzer = CircuitAnalyzer(backbone, meta_encoder, val_loader, device)
    z, logits, x, labels = analyzer.collect_all()
    centroids = analyzer.class_centroids(z, labels)
    dream = analyzer.activate_maximize(centroids[3])  # cat centroid dream
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
_STD  = torch.tensor([0.2470, 0.2435, 0.2616])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalised CIFAR-10 tensor → [0, 1]. Accepts [C,H,W] or [B,C,H,W]."""
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    if x.dim() == 4:
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    else:
        mean = mean[:, None, None]
        std  = std[:, None, None]
    return (x * std + mean).clamp(0, 1)


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation regularizer for [B, C, H, W] tensors."""
    return (
        (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() +
        (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    )


class CircuitAnalyzer:
    def __init__(
        self,
        backbone,
        meta_encoder,
        loader: DataLoader,
        device: torch.device,
    ):
        self.backbone     = backbone
        self.meta_encoder = meta_encoder
        self.loader       = loader
        self.device       = device

    # ------------------------------------------------------------------ #
    # Data collection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def collect_all(self, max_samples: int = 5000):
        """
        Collect circuit embeddings, output probabilities, images, and labels.

        Returns:
            z:      [N, D]          L2-normalised circuit embeddings (CPU)
            logits: [N, num_classes] softmax probabilities (CPU)
            x:      [N, 3, 32, 32]  normalised images (CPU)
            labels: [N]             integer class labels (CPU)
        """
        self.backbone.eval()
        self.meta_encoder.eval()

        all_z, all_logits, all_x, all_labels = [], [], [], []
        n = 0

        for batch in self.loader:
            imgs   = batch[0].to(self.device)
            labels = batch[-1]

            raw_logits, traj = self.backbone(imgs)
            z = self.meta_encoder(traj)

            all_z.append(z.cpu())
            all_logits.append(F.softmax(raw_logits, dim=-1).cpu())
            all_x.append(imgs.cpu())
            all_labels.append(labels.cpu())

            n += imgs.shape[0]
            if n >= max_samples:
                break

        z      = torch.cat(all_z,      0)[:max_samples]
        logits = torch.cat(all_logits, 0)[:max_samples]
        x      = torch.cat(all_x,      0)[:max_samples]
        labels = torch.cat(all_labels, 0)[:max_samples]
        return z, logits, x, labels

    # ------------------------------------------------------------------ #
    # Centroid computation
    # ------------------------------------------------------------------ #

    def class_centroids(self, z: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute the L2-normalised mean circuit embedding for each class.

        Returns:
            dict mapping class_index (int) → centroid tensor [D] (CPU)
        """
        centroids = {}
        for cls in range(10):
            mask = labels == cls
            c = z[mask].mean(0)
            centroids[cls] = F.normalize(c, dim=-1)
        return centroids

    # ------------------------------------------------------------------ #
    # Activation maximization
    # ------------------------------------------------------------------ #

    def activate_maximize(
        self,
        target_z:   torch.Tensor,
        n_steps:    int   = 512,
        lr:         float = 0.05,
        tv_weight:  float = 0.05,
        l2_weight:  float = 1e-3,
        blur_every: int   = 4,
        verbose:    bool  = False,
    ) -> torch.Tensor:
        """
        Optimise input pixels to minimise cosine distance to target_z.

        Gradient flows through the frozen backbone and meta-encoder back to
        the input image. TV and L2 regularisation suppress adversarial noise;
        periodic Gaussian blur further enforces spatial smoothness.

        Args:
            target_z:   [D] target circuit embedding (should be L2-normalised)
            n_steps:    optimisation steps
            lr:         Adam learning rate
            tv_weight:  total variation regularisation weight (dominant regulariser)
            l2_weight:  L2 image norm regularisation weight
            blur_every: apply 3×3 Gaussian blur every N steps (0 = disabled)
            verbose:    print loss every 100 steps

        Returns:
            [3, 32, 32] denormalised image in [0, 1] (CPU)
        """
        self.backbone.eval()
        self.meta_encoder.eval()

        x = torch.randn(1, 3, 32, 32, device=self.device) * 0.1
        x.requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)
        target    = F.normalize(target_z.to(self.device), dim=-1).unsqueeze(0)

        # 3×3 Gaussian blur kernel — built once, reused every blur_every steps
        _gk = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                            dtype=torch.float32, device=self.device) / 16.0
        _gk = _gk.view(1, 1, 3, 3).expand(3, 1, 3, 3)

        for step in range(n_steps):
            optimizer.zero_grad()

            _, traj = self.backbone(x)
            z = self.meta_encoder(traj)

            cos_loss = (1 - F.cosine_similarity(z, target)).mean()
            reg_loss = tv_weight * _total_variation(x) + l2_weight * (x ** 2).mean()
            (cos_loss + reg_loss).backward()
            optimizer.step()

            # Periodic Gaussian blur enforces spatial smoothness beyond TV alone
            if blur_every > 0 and (step + 1) % blur_every == 0:
                with torch.no_grad():
                    x.data.copy_(F.conv2d(x.data, _gk, padding=1, groups=3))

            if verbose and (step + 1) % 100 == 0:
                print(f"    step {step+1:4d}: cos={cos_loss.item():.4f}  "
                      f"reg={reg_loss.item():.5f}")

        with torch.no_grad():
            return denormalize(x.squeeze(0).detach()).cpu()

    # ------------------------------------------------------------------ #
    # k-nearest neighbour retrieval
    # ------------------------------------------------------------------ #

    def knn_circuit(
        self,
        query_z: torch.Tensor,
        all_z:   torch.Tensor,
        all_x:   torch.Tensor,
        k:       int = 5,
    ):
        """
        k-nearest neighbours of query_z in circuit space (cosine similarity).

        Args:
            query_z: [D]          single L2-normalised query embedding
            all_z:   [N, D]       corpus of L2-normalised embeddings
            all_x:   [N, 3, H, W] corresponding images

        Returns:
            (indices [k], images [k, 3, H, W], cosine_distances [k])
        """
        sims = all_z @ F.normalize(query_z, dim=-1)
        sims[(sims - 1.0).abs() < 1e-5] = -2.0   # exclude self
        topk = sims.topk(k).indices
        return topk, all_x[topk], 1 - sims[topk]

    def knn_output(
        self,
        query_logits: torch.Tensor,
        all_logits:   torch.Tensor,
        all_x:        torch.Tensor,
        k:            int = 5,
    ):
        """
        k-nearest neighbours in output (softmax probability) space.

        Cosine similarity on softmax distributions: two images are 'close' if
        the model is similarly uncertain between the same classes, not just if
        they share the same top-1 prediction.

        Args:
            query_logits: [C]    softmax probabilities for the query
            all_logits:   [N, C] corpus softmax probabilities
            all_x:        [N, 3, H, W] corresponding images

        Returns:
            (indices [k], images [k, 3, H, W], cosine_distances [k])
        """
        q      = F.normalize(query_logits, dim=-1)
        corpus = F.normalize(all_logits,   dim=-1)
        sims   = corpus @ q
        sims[(sims - 1.0).abs() < 1e-5] = -2.0   # exclude self
        topk   = sims.topk(k).indices
        return topk, all_x[topk], 1 - sims[topk]
