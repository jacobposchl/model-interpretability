"""
CIFAR-10 data loading for Phase 1 meta-encoder validation.

Standard (non-paired) data loading only. All pair computation happens
within-batch in the trainer — no class-label pairing is needed since
the training signal is derived from alignment profiles, not class labels.

Labels are retained in the dataset for evaluation purposes (class purity
analysis) but are NOT used in the training loss.
"""

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
# DataLoader factory
# --------------------------------------------------------------------------- #

def get_standard_loaders(data_dir: str, batch_size: int, num_workers: int = 4,
                         augment: bool = True, download: bool = True):
    """Standard loaders for Phase 1 training and evaluation."""
    train_ds = CIFAR10(
        root=data_dir, train=True,
        transform=get_train_transform(augment), download=download,
    )
    val_ds = CIFAR10(
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
