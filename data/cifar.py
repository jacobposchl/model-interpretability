"""
CIFAR-10 data loading with paired same-class sampling.

The paired sampler is the key non-standard component: for each item in the
dataset it returns a second image drawn uniformly at random from the same
class. These pairs are the inputs to the consistency loss — semantically
identical category, potentially very different surface properties.

Two dataset modes:
  PairedCIFAR10   — returns (img1, img2, label), used during CTLS training.
  StandardCIFAR10 — plain (img, label), used for baseline training and eval.
"""

import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_train_transform(augment: bool = True) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616)),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616)),
    ])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

class PairedCIFAR10(torch.utils.data.Dataset):
    """
    Returns (img1, img2, label) where img1 and img2 are from the same class.

    img1 is drawn at position idx; img2 is sampled randomly from the same
    class. Both images pass through the same transform, so augmentation
    produces different views — which is the intended behavior for the
    consistency loss.
    """

    def __init__(self, root: str, train: bool = True,
                 transform=None, download: bool = True):
        self.cifar = CIFAR10(root=root, train=train,
                             transform=transform, download=download)
        # Build class → index mapping without triggering transforms
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.cifar.targets):
            self.class_to_indices[label].append(idx)

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int):
        img1, label = self.cifar[idx]
        idx2 = random.choice(self.class_to_indices[label])
        img2, _ = self.cifar[idx2]
        return img1, img2, label


class StandardCIFAR10(torch.utils.data.Dataset):
    """Plain (img, label) wrapper — for baseline training and evaluation."""

    def __init__(self, root: str, train: bool = True,
                 transform=None, download: bool = True):
        self.cifar = CIFAR10(root=root, train=train,
                             transform=transform, download=download)

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int):
        return self.cifar[idx]


# --------------------------------------------------------------------------- #
# DataLoader factories
# --------------------------------------------------------------------------- #

def get_paired_loaders(data_dir: str, batch_size: int, num_workers: int = 4,
                       augment: bool = True, download: bool = True):
    """Paired loaders for CTLS training (Stage 2+)."""
    train_ds = PairedCIFAR10(
        root=data_dir, train=True,
        transform=get_train_transform(augment), download=download,
    )
    val_ds = PairedCIFAR10(
        root=data_dir, train=False,
        transform=get_val_transform(), download=download,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def get_standard_loaders(data_dir: str, batch_size: int, num_workers: int = 4,
                         augment: bool = True, download: bool = True):
    """Standard (non-paired) loaders for baseline training (Stage 1)."""
    train_ds = StandardCIFAR10(
        root=data_dir, train=True,
        transform=get_train_transform(augment), download=download,
    )
    val_ds = StandardCIFAR10(
        root=data_dir, train=False,
        transform=get_val_transform(), download=download,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
