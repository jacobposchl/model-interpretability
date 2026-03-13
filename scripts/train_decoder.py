"""
Train the CircuitDecoder post-hoc (Stage 6).

Loads a frozen CTLS checkpoint (backbone + meta-encoder) and trains a
CircuitDecoder to reconstruct input images from circuit embeddings z.

A second decoder is trained on the baseline checkpoint for comparison.
The baseline decoder measures the reconstruction quality achievable from
an unstructured circuit space — the delta in val MSE is the key metric.

Usage:
    python scripts/train_decoder.py --config configs/decoder.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from models.soft_mask import SoftMask
from models.backbone import CTLSBackbone
from models.meta_encoder import MetaEncoder
from models.decoder import CircuitDecoder
from data.cifar import get_standard_loaders


def load_frozen_encoder(config: dict, checkpoint_path: str, device: torch.device):
    """
    Load backbone + meta-encoder from a CTLS checkpoint and freeze both.

    SoftMask.temperature is not a buffer so it is not restored by load_state_dict.
    We initialise it from config['model']['soft_mask']['init_temperature'],
    which should be set to the final annealed temperature (0.1 for CTLS).
    """
    mcfg = config["model"]
    ecfg = mcfg["meta_encoder"]
    sm_temp = mcfg["soft_mask"]["init_temperature"]

    soft_mask = SoftMask(init_temperature=sm_temp)
    backbone = CTLSBackbone(
        arch=mcfg["arch"],
        num_classes=mcfg["num_classes"],
        soft_mask=soft_mask,
        pretrained=False,
    ).to(device)
    meta_encoder = MetaEncoder(
        layer_dims=backbone.layer_dims,
        hidden_dim=ecfg.get("hidden_dim", 256),
        embedding_dim=ecfg.get("embedding_dim", 64),
        encoder_type=ecfg.get("encoder_type", "mlp"),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    meta_encoder.load_state_dict(ckpt["meta_encoder_state"])
    print(f"  Loaded checkpoint: epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    backbone.eval()
    meta_encoder.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in meta_encoder.parameters():
        p.requires_grad_(False)

    return backbone, meta_encoder


def train_decoder(
    backbone,
    meta_encoder,
    train_loader,
    val_loader,
    config: dict,
    save_dir: Path,
    tag: str,
    device: torch.device,
) -> float:
    """
    Train a CircuitDecoder on embeddings from the given (frozen) encoder pair.

    Args:
        tag: label used in checkpoint filenames ('ctls' or 'baseline')

    Returns:
        Best validation MSE achieved.
    """
    tcfg = config["training"]
    dcfg = config["decoder"]
    lcfg = config["logging"]

    decoder = CircuitDecoder(
        embedding_dim=dcfg.get("embedding_dim", 64),
        hidden_channels=dcfg.get("hidden_channels", [256, 128, 64, 32]),
        input_spatial=dcfg.get("input_spatial", 4),
    ).to(device)

    epochs = tcfg["epochs"]
    optimizer = AdamW(
        decoder.parameters(),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    log_interval = lcfg.get("log_interval", 50)
    save_every = lcfg.get("save_every", 10)
    best_val_mse = float("inf")

    for epoch in range(epochs):
        decoder.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            with torch.no_grad():
                _, traj = backbone(x)
                z = meta_encoder(traj)

            x_hat = decoder(z)
            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  [{tag}] [{batch_idx+1}/{len(train_loader)}] "
                    f"mse={loss.item():.5f}"
                )

        scheduler.step()

        # Validation
        decoder.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                _, traj = backbone(x)
                z = meta_encoder(traj)
                x_hat = decoder(z)
                val_loss += F.mse_loss(x_hat, x).item()
                val_batches += 1

        avg_train = total_loss / n_batches
        avg_val = val_loss / val_batches
        print(
            f"Epoch {epoch+1:3d}/{epochs} [{tag}] | "
            f"train_mse={avg_train:.5f}  val_mse={avg_val:.5f}"
        )

        is_best = avg_val < best_val_mse
        if is_best:
            best_val_mse = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "val_mse": avg_val,
                    "decoder_state": decoder.state_dict(),
                    "config": config,
                },
                save_dir / f"decoder_{tag}_best.pt",
            )

        if (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "val_mse": avg_val,
                    "decoder_state": decoder.state_dict(),
                    "config": config,
                },
                save_dir / f"decoder_{tag}_epoch_{epoch+1}.pt",
            )

    print(f"[{tag}] Best val MSE: {best_val_mse:.5f}")
    return best_val_mse


def main():
    parser = argparse.ArgumentParser(description="Train circuit decoder (Stage 6)")
    parser.add_argument("--config", required=True, help="Path to decoder YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device:     {device}")

    save_dir = Path(config["logging"]["checkpoint_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    dcfg = config["data"]
    train_loader, val_loader = get_standard_loaders(
        data_dir=dcfg["data_dir"],
        batch_size=dcfg["batch_size"],
        num_workers=dcfg.get("num_workers", 4),
        augment=dcfg.get("augment", True),
        download=True,
    )

    # Train CTLS decoder
    print("\n=== Training CTLS decoder ===")
    backbone_ctls, meta_enc_ctls = load_frozen_encoder(
        config, config["training"]["ctls_checkpoint"], device
    )
    mse_ctls = train_decoder(
        backbone_ctls, meta_enc_ctls,
        train_loader, val_loader,
        config, save_dir, tag="ctls", device=device,
    )

    # Train baseline decoder for comparison
    print("\n=== Training Baseline decoder ===")
    backbone_base, meta_enc_base = load_frozen_encoder(
        config, config["training"]["baseline_checkpoint"], device
    )
    mse_base = train_decoder(
        backbone_base, meta_enc_base,
        train_loader, val_loader,
        config, save_dir, tag="baseline", device=device,
    )

    print("\n=== Final comparison ===")
    print(f"  CTLS decoder best val MSE:     {mse_ctls:.5f}")
    print(f"  Baseline decoder best val MSE: {mse_base:.5f}")
    print(f"  Delta (CTLS − baseline):       {mse_ctls - mse_base:+.5f}")


if __name__ == "__main__":
    main()
