"""
CTLS backbone: any standard architecture wrapped with forward hooks that
capture the activation trajectory T(x) = (h₁(x), ..., h_L(x)).

Design decisions:
  - Hooks fire on the output of each major block (BasicBlock for ResNet,
    encoder block for ViT). This gives L trajectory points where L equals
    the number of blocks.
  - ResNet spatial feature maps [B, C, H, W] are globally average-pooled to
    [B, C] so the trajectory is always a list of [B, D] tensors — compatible
    with both ResNet and ViT regardless of architecture.
  - ViT: CLS token [B, D] is extracted from each block's output [B, N+1, D].
  - Soft masking is applied to every captured activation before it is stored
    in the trajectory. The backbone's own computation is NOT changed — hooks
    are read-only with respect to the forward pass.
  - A dummy forward pass in __init__ discovers trajectory dims automatically,
    so the MetaEncoder can be constructed with the correct input sizes.

Supported architectures (via torchvision):
  ResNet family: resnet18, resnet34, resnet50, resnet101
  ViT family:    vit_b_16, vit_s_16 (requires timm if not in torchvision)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from models.soft_mask import SoftMask


class CTLSBackbone(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        soft_mask: SoftMask,
        pretrained: bool = False,
    ):
        super().__init__()
        self.soft_mask = soft_mask
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

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns:
            logits:     [B, num_classes]
            trajectory: list of L tensors, each [B, D_l]
        """
        self._trajectory = []
        logits = self.model(x)
        return logits, list(self._trajectory)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

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
                # Some ViT blocks return (tensor, ...) tuples
                tensor = output[0]
            else:
                tensor = output

            if tensor.dim() == 4:
                # ResNet: [B, C, H, W] → global average pool → [B, C]
                pooled = tensor.mean(dim=[2, 3])
            elif tensor.dim() == 3:
                # ViT: [B, N+1, D] → CLS token → [B, D]
                pooled = tensor[:, 0]
            else:
                pooled = tensor  # already [B, D]

            masked = self.soft_mask(pooled)
            self._trajectory.append(masked)

        return hook

    def _discover_dims(self, arch: str) -> list[int]:
        """Run a tiny dummy forward pass to learn trajectory tensor shapes."""
        # CIFAR-10 uses 32×32 inputs
        dummy_size = (1, 3, 32, 32) if "cifar" in arch or True else (1, 3, 224, 224)
        dummy = torch.zeros(dummy_size)
        with torch.no_grad():
            _, traj = self.forward(dummy)
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

    # For CIFAR-10 (32×32), the default ResNet stem (stride-2 conv + maxpool)
    # aggressively downsamples. Replace with a small 3×3 conv stem.
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
