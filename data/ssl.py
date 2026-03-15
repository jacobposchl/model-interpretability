"""
SSL data loading: multi-view augmentation and CIFAR-100 transfer datasets.

Three dataset modes:
  MultiViewDataset    — returns (view1, view2, idx); no label. The idx is the
                        dataset position, used to update the embedding bank in
                        CTLS-SSL Phase 2 neighbor mining.
  CIFAR100ForTransfer — labeled CIFAR-100 split by superclass into 'base' (first
                        15 superclasses) and 'novel' (last 5 superclasses). Used
                        for few-shot transfer evaluation.

SSL augmentation is intentionally stronger than the CTLS supervised augmentation:
  - RandomResizedCrop with scale=(0.2, 1.0) forces the model to be invariant to
    aggressive scale and aspect ratio changes — a key SimCLR signal source.
  - GaussianBlur promotes invariance to fine-grained texture details.
  - RandomGrayscale promotes cross-channel invariance.
  - Stronger ColorJitter (0.4 vs 0.2) forces color invariance.

The val transform stays identical to cifar.py's get_val_transform() so that
evaluation metrics are comparable across supervised and SSL experiments.
"""

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from data.cifar import get_val_transform


# --------------------------------------------------------------------------- #
# CIFAR-100 fine → coarse (superclass) mapping
# 100 entries: index i → superclass label for fine class i
# Superclass names (0-19):
#   0: aquatic_mammals, 1: fish, 2: flowers, 3: food_containers,
#   4: fruit_and_vegetables, 5: household_electrical_devices,
#   6: household_furniture, 7: insects, 8: large_carnivores,
#   9: large_man-made_outdoor, 10: large_natural_outdoor,
#   11: large_omnivores_herbivores, 12: medium_mammals,
#   13: non-insect_invertebrates, 14: people, 15: reptiles,
#   16: small_mammals, 17: trees, 18: vehicles_1, 19: vehicles_2
# --------------------------------------------------------------------------- #

_CIFAR100_FINE_TO_COARSE = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,   # 0-9
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,   # 10-19
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,   # 20-29
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,   # 30-39
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,   # 40-49
   16,  4, 17,  4,  2,  0, 17,  4, 18, 17,   # 50-59
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,   # 60-69
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,   # 70-79
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,   # 80-89
   18,  1,  2, 15,  6,  0, 17,  8, 14, 13,   # 90-99
]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_ssl_augmentation(img_size: int = 32) -> transforms.Compose:
    """
    Strong SSL augmentation for CIFAR-scale images.

    Designed to be the training transform for MultiViewDataset. Both views of
    the same image pass through this independently, producing two meaningfully
    different crops that share high-level semantics — the invariance target for
    self-supervised contrastive learning.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

class MultiViewDataset(torch.utils.data.Dataset):
    """
    Wraps a torchvision image dataset and returns two independently-augmented
    views of the same image along with its dataset position index.

    Returns (view1, view2, idx) — no label is included, keeping the interface
    self-supervised. The idx is the integer dataset position, used by
    EmbeddingBank.update() in CTLS-SSL Phase 2 to maintain per-sample
    embeddings for nearest-neighbor mining.

    For linear probe and KNN evaluation, use get_multiview_loaders() which
    returns a separate standard labeled val_loader.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        augmentation=None,
        dataset_name: str = "cifar10",
        download: bool = True,
    ):
        if dataset_name == "cifar10":
            self.dataset = CIFAR10(
                root=root, train=train, transform=None, download=download
            )
        elif dataset_name == "cifar100":
            self.dataset = CIFAR100(
                root=root, train=train, transform=None, download=download
            )
        else:
            raise ValueError(f"Unsupported dataset_name: '{dataset_name}'. Use 'cifar10' or 'cifar100'.")

        self.augmentation = augmentation or get_ssl_augmentation()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]   # PIL image, label discarded
        view1 = self.augmentation(img)
        view2 = self.augmentation(img)
        return view1, view2, idx


class CIFAR100ForTransfer(torch.utils.data.Dataset):
    """
    Labeled CIFAR-100 for few-shot transfer evaluation, split by superclass.

    CIFAR-100 has 20 superclasses (0–19), each containing 5 fine classes (100 total).
    The split is fixed:
      'base'  — superclasses 0–14 (75 fine classes): used during analysis only
      'novel' — superclasses 15–19 (25 fine classes): the few-shot evaluation target
      'all'   — all 100 fine classes

    Semantic rationale for the base/novel split:
      Novel superclasses were chosen to test the CTLS-SSL circuit scaffold hypothesis:
        15 (reptiles)    — medium semantic distance to CIFAR-10
        16 (small_mammals) — close (analogous to cat/dog/deer)
        17 (trees)       — distant (no CIFAR-10 analog)
        18 (vehicles_1)  — close (analogous to automobile/truck/airplane)
        19 (vehicles_2)  — medium (partial vehicle overlap)

    Attributes:
        class_to_indices (dict[int, list[int]]): fine_label → list of sample indices.
            Used by EpisodeSampler for N-way K-shot episode construction.
    """

    _NOVEL_SUPERCLASSES = {15, 16, 17, 18, 19}
    _BASE_SUPERCLASSES = set(range(15))

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        split: str = "novel",
        download: bool = True,
    ):
        if split not in ("all", "base", "novel"):
            raise ValueError(f"split must be 'all', 'base', or 'novel', got '{split}'")

        base_ds = CIFAR100(root=root, train=train, transform=None, download=download)
        self.transform = transform or get_val_transform()

        if split == "all":
            allowed_coarse = set(range(20))
        elif split == "base":
            allowed_coarse = self._BASE_SUPERCLASSES
        else:  # novel
            allowed_coarse = self._NOVEL_SUPERCLASSES

        # Filter indices that belong to the requested superclass split
        self._indices: list[int] = []
        self._labels: list[int] = []
        for i, fine_label in enumerate(base_ds.targets):
            if _CIFAR100_FINE_TO_COARSE[fine_label] in allowed_coarse:
                self._indices.append(i)
                self._labels.append(fine_label)

        self._base_ds = base_ds

        # Build class_to_indices over the filtered positions
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for pos, fine_label in enumerate(self._labels):
            self.class_to_indices[fine_label].append(pos)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        base_idx = self._indices[idx]
        img, _ = self._base_ds[base_idx]   # PIL image
        label = self._labels[idx]
        return self.transform(img), label


# --------------------------------------------------------------------------- #
# DataLoader factories
# --------------------------------------------------------------------------- #

def get_multiview_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    download: bool = True,
):
    """
    Returns (train_loader, val_loader) for SSL training.

    train_loader: yields (view1, view2, idx) — no labels, SSL-style.
    val_loader:   yields (img, label) — standard CIFAR-10 validation set
                  with val_transform, for linear probe and KNN evaluation.
    """
    from data.cifar import StandardCIFAR10

    train_ds = MultiViewDataset(
        root=data_dir,
        train=True,
        augmentation=get_ssl_augmentation(),
        dataset_name="cifar10",
        download=download,
    )
    val_ds = StandardCIFAR10(
        root=data_dir,
        train=False,
        transform=get_val_transform(),
        download=download,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # NT-Xent needs consistent batch size
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_cifar100_loaders(
    data_dir: str,
    batch_size: int,
    split: str = "novel",
    num_workers: int = 4,
    download: bool = True,
) -> DataLoader:
    """Labeled CIFAR-100 loader for few-shot transfer evaluation."""
    ds = CIFAR100ForTransfer(
        root=data_dir,
        train=False,
        transform=get_val_transform(),
        split=split,
        download=download,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
