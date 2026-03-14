"""
Circuit analysis utilities for Stage 6 (circuit visualization) and
Stage 7 (trajectory divergence analysis).

Stage 6 — two experiments using the frozen CTLS checkpoint:
  1. Circuit Nearest Neighbors — retrieve real images closest to any target
     point in circuit space (centroid, interpolation step, confusion zone).
  2. Circuit vs Output Neighbors + GradCAM — compare k-NN in circuit space
     vs softmax-probability space; GradCAM localises which spatial regions
     drive the circuit similarity to a target class.

Stage 7 — trajectory divergence (novel diagnostic):
  Uses the raw 8-layer activation trajectory (before MetaEncoder compression)
  to answer *at which layer* a specific image's processing diverges from the
  expected path for its true class.  No additional training required.
  Key quantities:
    - divergence_curve:        per-layer cosine distance to true-class centroid
    - layer_class_similarities: per-layer cosine similarity to all 10 classes
    - defection layer:          first layer where sim(pred) > sim(true)

Note on activation maximization:
  The backbone hook globally-average-pools before storing each trajectory
  step.  This makes d(traj)/d(x[h,w]) spatially uniform — pixel optimisation
  converges to uniform gray.  Nearest-neighbor retrieval and GradCAM are
  used instead.

Usage:
    from evaluation.circuit_analysis import CircuitAnalyzer, denormalize, CIFAR10_CLASSES
    analyzer = CircuitAnalyzer(backbone, meta_encoder, val_loader, device)
    # Stage 6
    z, logits, x, labels = analyzer.collect_all()
    centroids = analyzer.class_centroids(z, labels)
    # Stage 7
    trajs, logits, labels = analyzer.collect_trajectories()
    lc = analyzer.layer_class_centroids(trajs, labels)
    curve = analyzer.trajectory_divergence_curve([trajs[l][i] for l in range(8)], true_cls, lc)
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
            z:      [N, D]           L2-normalised circuit embeddings (CPU)
            logits: [N, num_classes]  softmax probabilities (CPU)
            x:      [N, 3, 32, 32]   normalised images (CPU)
            labels: [N]              integer class labels (CPU)
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
    # Nearest-neighbour retrieval
    # ------------------------------------------------------------------ #

    def nearest_to_target(
        self,
        target_z:   torch.Tensor,
        z_all:      torch.Tensor,
        x_all:      torch.Tensor,
        labels_all: torch.Tensor,
        k:          int = 5,
    ) -> tuple:
        """
        k real images from the corpus closest (cosine) to an arbitrary target
        in circuit space.  Unlike knn_circuit, target_z need not be in the
        corpus — use this for centroids, interpolation points, etc.

        Returns:
            (indices [k], images [k, C, H, W], labels [k], cosine_sims [k])
        """
        target = F.normalize(target_z.cpu(), dim=-1)
        sims   = z_all @ target
        topk   = sims.topk(k).indices
        return topk, x_all[topk], labels_all[topk], sims[topk]

    def knn_circuit(
        self,
        query_z: torch.Tensor,
        all_z:   torch.Tensor,
        all_x:   torch.Tensor,
        k:       int = 5,
    ) -> tuple:
        """
        k-nearest neighbours of an in-corpus query in circuit space.
        Excludes exact self-match.

        Returns:
            (indices [k], images [k, 3, H, W], cosine_distances [k])
        """
        sims = all_z @ F.normalize(query_z, dim=-1)
        sims[(sims - 1.0).abs() < 1e-5] = -2.0
        topk = sims.topk(k).indices
        return topk, all_x[topk], 1 - sims[topk]

    def knn_output(
        self,
        query_logits: torch.Tensor,
        all_logits:   torch.Tensor,
        all_x:        torch.Tensor,
        k:            int = 5,
    ) -> tuple:
        """
        k-nearest neighbours in output (softmax probability) space.
        Cosine similarity on softmax distributions.

        Returns:
            (indices [k], images [k, 3, H, W], cosine_distances [k])
        """
        q      = F.normalize(query_logits, dim=-1)
        corpus = F.normalize(all_logits,   dim=-1)
        sims   = corpus @ q
        sims[(sims - 1.0).abs() < 1e-5] = -2.0
        topk   = sims.topk(k).indices
        return topk, all_x[topk], 1 - sims[topk]

    # ------------------------------------------------------------------ #
    # Trajectory divergence analysis (Stage 7)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def collect_trajectories(self, max_samples: int = 10000):
        """
        Collect raw per-layer activations (before MetaEncoder) for all val samples.

        Returns:
            trajs:  list of L tensors, each [N, D_l]  — per-layer activations (CPU)
            logits: [N, num_classes]                   — softmax probabilities (CPU)
            labels: [N]                                — integer class labels (CPU)
        """
        self.backbone.eval()

        all_trajs: list[list] = None   # type: ignore[assignment]
        all_logits, all_labels = [], []
        n = 0

        for batch in self.loader:
            imgs   = batch[0].to(self.device)
            labels = batch[-1]

            raw_logits, traj = self.backbone(imgs)

            if all_trajs is None:
                all_trajs = [[] for _ in range(len(traj))]
            for l, h in enumerate(traj):
                all_trajs[l].append(h.cpu())

            all_logits.append(F.softmax(raw_logits, dim=-1).cpu())
            all_labels.append(labels.cpu())

            n += imgs.shape[0]
            if n >= max_samples:
                break

        trajs  = [torch.cat(all_trajs[l], 0)[:max_samples] for l in range(len(all_trajs))]
        logits = torch.cat(all_logits, 0)[:max_samples]
        labels = torch.cat(all_labels, 0)[:max_samples]
        return trajs, logits, labels

    def layer_class_centroids(
        self,
        trajs:  list[torch.Tensor],
        labels: torch.Tensor,
    ) -> list[dict]:
        """
        Per-layer, per-class L2-normalised centroids in raw activation space.

        Returns:
            List of L dicts, each mapping class_idx (int) → [D_l] centroid (CPU)
        """
        centroids = []
        for l, h_all in enumerate(trajs):
            layer_cents = {}
            for cls in range(10):
                mask = labels == cls
                c = h_all[mask].mean(0)
                layer_cents[cls] = F.normalize(c, dim=-1)
            centroids.append(layer_cents)
        return centroids

    def trajectory_divergence_curve(
        self,
        traj:        list[torch.Tensor],
        true_cls:    int,
        layer_cents: list[dict],
    ) -> torch.Tensor:
        """
        Per-layer cosine distance to the true-class centroid for a single image.

        Args:
            traj:        list of L tensors, each [D_l]  (single image, CPU)
            true_cls:    ground-truth class index
            layer_cents: as returned by layer_class_centroids

        Returns:
            [L] cosine distances  (0 = identical, 1 = orthogonal)
        """
        dists = []
        for l, h in enumerate(traj):
            cent  = layer_cents[l][true_cls]
            h_n   = F.normalize(h, dim=-1)
            dists.append(1.0 - (h_n * cent).sum())
        return torch.stack(dists)

    def layer_class_similarities(
        self,
        h:             torch.Tensor,
        layer_cents_l: dict,
    ) -> torch.Tensor:
        """
        Cosine similarity of a single layer activation to all 10 class centroids.

        Args:
            h:             [D_l]  single layer activation (CPU)
            layer_cents_l: dict mapping class_idx → [D_l] centroid for that layer

        Returns:
            [10] cosine similarities
        """
        h_n = F.normalize(h, dim=-1)
        return torch.stack([
            (h_n * layer_cents_l[c]).sum() for c in range(10)
        ])

    # ------------------------------------------------------------------ #
    # GradCAM on circuit similarity
    # ------------------------------------------------------------------ #

    def gradcam(
        self,
        x_img:    torch.Tensor,
        target_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        GradCAM of cosine similarity to target_z w.r.t. the last spatial
        feature map in the backbone.

        Although the backbone trajectory is globally average-pooled, the
        gradient of the loss w.r.t. the PRE-POOL spatial feature map still
        varies per channel (alpha_c differs because each channel contributes
        differently to the pooled trajectory → z → cosine similarity).
        Weighting the spatial map by alpha_c and summing over channels
        reveals where those channels are most active — i.e., which spatial
        regions are most circuit-relevant for the target class.

        This is standard Grad-CAM applied to the last ResNet block before
        the global-average-pool in the trajectory hook.

        Args:
            x_img:    [3, 32, 32] normalised input image
            target_z: [D] target circuit embedding (L2-normalised)

        Returns:
            [32, 32] heatmap in [0, 1] (CPU)
        """
        # Leaf tensor with grad so the spatial feature map enters the graph
        x      = x_img.detach().unsqueeze(0).to(self.device).requires_grad_(True)
        target = F.normalize(target_z.to(self.device), dim=-1).unsqueeze(0)

        saved = {}

        def _fwd(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            saved['feat'] = t
            # Hook fires during backward with d(loss)/d(t)
            t.register_hook(lambda g: saved.__setitem__('grad', g))

        hook = self.backbone._hook_modules[-1].register_forward_hook(_fwd)
        try:
            self.backbone.eval()
            self.meta_encoder.eval()

            _, traj = self.backbone(x)
            z       = self.meta_encoder(traj)
            F.cosine_similarity(z, target).mean().backward()

            feat = saved.get('feat')
            grad = saved.get('grad')

            if feat is None or grad is None or feat.dim() != 4:
                # ViT or unexpected shape — return a neutral uniform map
                return torch.ones(32, 32) * 0.5

            # alpha_c = mean gradient per channel  [1, C, 1, 1]
            weights = grad.mean(dim=[2, 3], keepdim=True)
            # Weighted sum over channels, ReLU to keep positive contributions
            cam = F.relu((weights * feat).sum(dim=1).squeeze(0))  # [H, W]
            # Upsample to input resolution
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(32, 32),
                mode='bilinear',
                align_corners=False,
            ).squeeze()
            mn, mx = cam.min(), cam.max()
            if mx > mn:
                cam = (cam - mn) / (mx - mn)
            return cam.detach().cpu()
        finally:
            hook.remove()
            self.backbone.zero_grad()
            self.meta_encoder.zero_grad()
