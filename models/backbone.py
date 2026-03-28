"""
Frozen backbone: a pretrained network wrapped with forward hooks that capture
the activation trajectory T(x) = (h_1(x), ..., h_L(x)).

Design decisions:
  - The backbone is FROZEN throughout Phase 1. No gradients flow through it.
    It is a read-only feature extractor.
  - Hooks fire on the output of each major block (BasicBlock for ResNet,
    encoder block for ViT). This gives L trajectory points where L equals
    the number of blocks.
  - ResNet spatial feature maps [B, C, H, W] are pooled to [B, C] via
    configurable pooling strategy (GAP, max, or top-k).
  - ViT: CLS token [B, D] is extracted from each block's output [B, N+1, D].
  - Each layer output is L2-normalized before storage, ensuring every layer
    contributes comparably to trajectory distances regardless of activation
    magnitude at different depths.
  - All captured activations are detached (stop-gradient) since the backbone
    is not being trained.
  - A dummy forward pass in __init__ discovers trajectory dims automatically,
    so the MetaEncoder can be constructed with the correct input sizes.

Supported architectures (via torchvision):
  ResNet family: resnet18, resnet34, resnet50, resnet101
  ViT family:    vit_b_16, vit_s_16 (requires timm)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class FrozenBackbone(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool = True,
        pool_mode: str = "gap",
    ):
        """
        Args:
            arch:        Architecture name (resnet18, resnet34, etc.)
            num_classes: Number of output classes (needed for model construction).
            pretrained:  Whether to load pretrained weights.
            pool_mode:   Pooling strategy for spatial feature maps.
                         One of 'gap' (global average), 'max' (global max),
                         'topk' (mean of top-10% activations by magnitude).
        """
        super().__init__()

        valid_pools = ("gap", "max", "topk")
        if pool_mode not in valid_pools:
            raise ValueError(f"pool_mode must be one of {valid_pools}, got '{pool_mode}'")
        self.pool_mode = pool_mode

        self._hook_handles: list = []
        self._trajectory: list[torch.Tensor] = []

        if arch.startswith("resnet"):
            self.model, self._hook_modules = _build_resnet(arch, num_classes, pretrained)
        elif arch.startswith("vit"):
            self.model, self._hook_modules = _build_vit(arch, num_classes, pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {arch}. Use resnet* or vit*.")

        self._register_hooks()

        # Discover trajectory dims via a dummy pass (used by MetaEncoder init)
        self.layer_dims: list[int] = self._discover_dims(arch)

        # Freeze all parameters — backbone is read-only
        self.requires_grad_(False)
        self.eval()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns:
            trajectory: list of L tensors, each [B, D_l], L2-normalized, detached
        """
        self._trajectory = []
        with torch.no_grad():
            self.model(x)
        return list(self._trajectory)

    def train(self, mode: bool = True):
        """No-op: backbone stays in eval mode permanently."""
        return self

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _register_hooks(self):
        for module in self._hook_modules:
            handle = module.register_forward_hook(self._make_hook())
            self._hook_handles.append(handle)

    def _make_hook(self):
        def hook(module, input, output):
            # Normalise output to [B, D]
            if isinstance(output, (tuple, list)):
                tensor = output[0]
            else:
                tensor = output

            if tensor.dim() == 4:
                pooled = self._pool_spatial(tensor)
            elif tensor.dim() == 3:
                # ViT: [B, N+1, D] -> CLS token -> [B, D]
                pooled = tensor[:, 0]
            else:
                pooled = tensor

            # L2-normalize and detach (stop-gradient)
            normed = F.normalize(pooled, dim=-1).detach()
            self._trajectory.append(normed)

        return hook

    def _pool_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pool spatial feature maps [B, C, H, W] -> [B, C]."""
        if self.pool_mode == "gap":
            return tensor.mean(dim=[2, 3])
        elif self.pool_mode == "max":
            return tensor.amax(dim=[2, 3])
        else:  # topk
            B, C, H, W = tensor.shape
            flat = tensor.flatten(2)  # [B, C, H*W]
            k = max(1, math.ceil(0.1 * H * W))
            topk_vals = flat.topk(k, dim=2).values  # [B, C, k]
            return topk_vals.mean(dim=2)  # [B, C]

    def _discover_dims(self, arch: str) -> list[int]:
        """Run a tiny dummy forward pass to learn trajectory tensor shapes."""
        dummy_size = (1, 3, 32, 32)
        dummy = torch.zeros(dummy_size)
        with torch.no_grad():
            traj = self.forward(dummy)
        return [h.shape[-1] for h in traj]


# --------------------------------------------------------------------------- #
# Architecture builders
# --------------------------------------------------------------------------- #

def _build_resnet(
    arch: str, num_classes: int, pretrained: bool
) -> tuple[nn.Module, list[nn.Module]]:
    weights = "IMAGENET1K_V1" if pretrained else None
    model: nn.Module = getattr(tvm, arch)(weights=weights)

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # For CIFAR-10 (32x32), the default ResNet stem (stride-2 conv + maxpool)
    # aggressively downsamples. Replace with a small 3x3 conv stem.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Collect all BasicBlock / Bottleneck modules as hook targets
    hook_targets: list[nn.Module] = []
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block in layer:
            hook_targets.append(block)

    return model, hook_targets


def _build_vit(
    arch: str, num_classes: int, pretrained: bool
) -> tuple[nn.Module, list[nn.Module]]:
    try:
        import timm
        model = timm.create_model(
            arch, pretrained=pretrained, num_classes=num_classes, img_size=32
        )
        hook_targets = list(model.blocks)
    except ImportError:
        raise ImportError(
            "timm is required for ViT architectures. "
            "Install with: pip install timm"
        )
    return model, hook_targets
