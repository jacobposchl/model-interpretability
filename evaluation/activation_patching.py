"""
Activation Patching — Step 2 of the refinement roadmap.

Provides ground-truth circuit similarity between pairs of inputs via causal
intervention: for each layer l, we replace x_b's activations at layer l with
x_a's activations and measure how much the final output changes. Low output
change → the two inputs used similar computation at layer l → evidence of a
shared circuit.

CircuitSim(x_a, x_b) aggregates this signal across all layers:
    influence_l = KL(softmax(logits_b_clean) ‖ softmax(logits_b_patched_at_l))
    CircuitSim  = 1 − mean_l(influence_l / max_l(influence_l))

High CircuitSim (close to 1) = shared circuit; low (close to 0) = divergent.

Implementation note: patching is done by registering a temporary forward hook
on backbone._hook_modules[l] that returns patch_value as the module output.
PyTorch uses the hook's return value instead of the actual output, so all
downstream layers see the patched activation. The backbone's read-only
trajectory hooks still fire (they capture the pre-patch actual output), but we
discard _trajectory from patched runs — we only need final logits.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class ActivationPatcher:
    """
    Ground-truth circuit similarity via activation patching.

    Args:
        backbone: CTLSBackbone instance (eval mode assumed during use).
        device:   torch.device for computation.
    """

    def __init__(self, backbone, device):
        self.backbone = backbone
        self.device = device
        self._n_layers = len(backbone._hook_modules)

    # ------------------------------------------------------------------ #
    # Core methods
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _get_clean(self, x: torch.Tensor):
        """Run a clean forward pass. Returns (logits, trajectory)."""
        logits, traj = self.backbone(x)
        return logits, traj

    @torch.no_grad()
    def _patched_logits(
        self, x_b: torch.Tensor, layer_idx: int, patch_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Run x_b with layer `layer_idx` output replaced by `patch_value`.

        A temporary forward hook is registered on backbone._hook_modules[layer_idx].
        Returning a non-None value from a PyTorch forward hook replaces the
        module's output, so all subsequent layers process patch_value instead.
        The hook is removed in a finally block to guarantee cleanup.
        """
        def _replace(module, input, output):
            return patch_value

        module = self.backbone._hook_modules[layer_idx]
        handle = module.register_forward_hook(_replace)
        try:
            logits, _ = self.backbone(x_b)
        finally:
            handle.remove()
        return logits

    def compute_pair_similarity(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> tuple[float, np.ndarray]:
        """
        Compute CircuitSim(x_a, x_b) via per-layer activation patching.

        Args:
            x_a: [1, C, H, W] tensor on device
            x_b: [1, C, H, W] tensor on device

        Returns:
            circuit_sim:         Scalar in [0, 1]. High = shared circuit.
            per_layer_kl:        np.ndarray of shape [L] with raw (unnormalised)
                                 KL divergence per layer. Useful for per-layer
                                 influence plots.
        """
        # Get x_a trajectory (the values we will patch into x_b)
        _, traj_a = self._get_clean(x_a)

        # Get x_b baseline logits
        logits_b_clean, _ = self._get_clean(x_b)
        p_b_clean = F.softmax(logits_b_clean, dim=-1)  # [1, C]

        per_layer_kl = []
        for l in range(self._n_layers):
            logits_b_patched = self._patched_logits(x_b, l, traj_a[l])
            log_p_patched = F.log_softmax(logits_b_patched, dim=-1)
            # KL(p_clean ‖ p_patched): measures how much the output distribution
            # changed when we swapped layer l's activations.
            kl = F.kl_div(log_p_patched, p_b_clean, reduction="sum").item()
            kl = max(kl, 0.0)  # KL ≥ 0; numerical errors can push it slightly below
            per_layer_kl.append(kl)

        per_layer_kl = np.array(per_layer_kl, dtype=np.float32)

        # Normalise by the maximum influence so all pairs are on [0, 1].
        max_kl = per_layer_kl.max()
        if max_kl > 1e-8:
            normalised = per_layer_kl / max_kl
        else:
            # No layer had any influence — activations are identical.
            normalised = np.zeros_like(per_layer_kl)

        circuit_sim = float(1.0 - normalised.mean())
        return circuit_sim, per_layer_kl

    # ------------------------------------------------------------------ #
    # Batch runner
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        desc: str = "Patching",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run compute_pair_similarity over a list of image pairs.

        Args:
            pairs: List of (x_a, x_b) tuples, each a [1, C, H, W] tensor on device.
            desc:  Label for the tqdm progress bar.

        Returns:
            sim_scores:         np.ndarray [N] of CircuitSim values.
            per_layer_kl_all:   np.ndarray [N, L] of raw per-layer KL divergences.
        """
        self.backbone.eval()

        sim_scores = []
        per_layer_kl_all = []

        for x_a, x_b in tqdm(pairs, desc=desc):
            sim, kl = self.compute_pair_similarity(x_a, x_b)
            sim_scores.append(sim)
            per_layer_kl_all.append(kl)

        return np.array(sim_scores, dtype=np.float32), np.array(per_layer_kl_all, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pair sampling utilities
# --------------------------------------------------------------------------- #

def sample_pairs(
    dataset,
    n_same: int = 500,
    n_diff: int = 500,
    seed: int = 42,
) -> tuple[list, list, np.ndarray]:
    """
    Sample same-class and different-class index pairs from a dataset.

    Args:
        dataset:  A dataset where dataset[i] returns (image, label).
        n_same:   Number of same-class pairs to sample.
        n_diff:   Number of different-class pairs to sample.
        seed:     Random seed for reproducibility.

    Returns:
        pairs:   List of (idx_a, idx_b) tuples (length n_same + n_diff).
        labels:  np.ndarray [N] with 1 = same-class, 0 = different-class.
        classes: np.ndarray [N] with the class of x_a for each pair.
    """
    rng = np.random.default_rng(seed)

    # Build class → index mapping without triggering transforms
    class_to_indices: dict[int, list[int]] = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        label = int(label)
        class_to_indices.setdefault(label, []).append(i)

    all_classes = sorted(class_to_indices.keys())
    pairs = []
    pair_labels = []
    pair_classes = []

    # Same-class pairs
    per_class_same = max(1, n_same // len(all_classes))
    for cls in all_classes:
        indices = class_to_indices[cls]
        if len(indices) < 2:
            continue
        for _ in range(per_class_same):
            a, b = rng.choice(indices, size=2, replace=False)
            pairs.append((int(a), int(b)))
            pair_labels.append(1)
            pair_classes.append(cls)
        if len(pairs) >= n_same:
            break

    # Different-class pairs
    per_class_diff = max(1, n_diff // len(all_classes))
    for cls in all_classes:
        other_classes = [c for c in all_classes if c != cls]
        indices_a = class_to_indices[cls]
        for _ in range(per_class_diff):
            other_cls = rng.choice(other_classes)
            indices_b = class_to_indices[other_cls]
            a = int(rng.choice(indices_a))
            b = int(rng.choice(indices_b))
            pairs.append((a, b))
            pair_labels.append(0)
            pair_classes.append(cls)
        if len(pairs) >= n_same + n_diff:
            break

    return pairs, np.array(pair_labels, dtype=np.int32), np.array(pair_classes, dtype=np.int32)
