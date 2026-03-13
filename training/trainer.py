"""
CTLS Trainer.

Handles the full training loop for both the baseline (Stage 1) and the full
CTLS objective (Stage 2+). Config-driven — all hyperparameters come from the
YAML config dict.

Training flow for CTLS (paired mode):
  1. Load paired batch (x1, x2, labels) from PairedCIFAR10.
  2. Concatenate [x1, x2] along batch dim → single forward pass → split
     trajectory and logits back out. One forward pass = full efficiency.
  3. Task loss:      cross_entropy(logits1, labels) + cross_entropy(logits2, labels)
  4. Consistency:    ℒ_cons(traj1, traj2)   — depth-weighted per-layer cosine distance
  5. SupCon:         ℒ_supcon(z_all, labels) — supervised contrastive on circuit embeddings;
                     attracts same-class embeddings, repels different-class embeddings,
                     preventing degenerate trajectory collapse.
  6. Total loss: ℒ_task + λ * ℒ_cons + μ * ℒ_supcon
  7. Update λ (linear warmup) and τ (cosine annealing) each epoch.

For baseline mode (paired=False), x2 and both auxiliary losses are ignored.
"""

import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.soft_mask import SoftMask
from models.backbone import CTLSBackbone
from models.meta_encoder import MetaEncoder
from losses.consistency import CircuitConsistencyLoss
from losses.contrastive import SupConLoss
from data.cifar import get_paired_loaders, get_standard_loaders
from training.schedulers import LambdaScheduler, TauScheduler


class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_models()
        self._build_data()
        self._build_optimizers()
        self._build_losses()
        self._build_schedulers()

        self.checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _build_models(self):
        cfg = self.cfg
        mcfg = cfg["model"]

        # SoftMask must stay on CPU until after CTLSBackbone.__init__ completes,
        # because _discover_dims runs a CPU dummy forward pass. The backbone
        # registers soft_mask as a submodule, so .to(device) below moves both.
        self.soft_mask = SoftMask(
            init_temperature=cfg["training"]["temperature"]["init"]
        )

        self.backbone = CTLSBackbone(
            arch=mcfg["arch"],
            num_classes=mcfg["num_classes"],
            soft_mask=self.soft_mask,
            pretrained=mcfg.get("pretrained", False),
        ).to(self.device)

        ecfg = mcfg["meta_encoder"]
        self.meta_encoder = MetaEncoder(
            layer_dims=self.backbone.layer_dims,
            hidden_dim=ecfg.get("hidden_dim", 256),
            embedding_dim=ecfg.get("embedding_dim", 64),
            encoder_type=ecfg.get("encoder_type", "mlp"),
        ).to(self.device)

    def _build_data(self):
        dcfg = self.cfg["data"]
        paired = self.cfg["training"].get("paired", True)

        if paired:
            self.train_loader, self.val_loader = get_paired_loaders(
                data_dir=dcfg["data_dir"],
                batch_size=dcfg["batch_size"],
                num_workers=dcfg.get("num_workers", 4),
                augment=dcfg.get("augment", True),
                download=True,
            )
        else:
            self.train_loader, self.val_loader = get_standard_loaders(
                data_dir=dcfg["data_dir"],
                batch_size=dcfg["batch_size"],
                num_workers=dcfg.get("num_workers", 4),
                augment=dcfg.get("augment", True),
                download=True,
            )
        self.paired = paired

    def _build_optimizers(self):
        tcfg = self.cfg["training"]
        lr = float(tcfg.get("lr", 1e-3))
        params = list(self.backbone.parameters()) + list(self.meta_encoder.parameters())
        self.optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg["epochs"],
            eta_min=lr * 0.01,
        )

    def _build_losses(self):
        tcfg = self.cfg["training"]
        ccfg = tcfg.get("consistency_loss", {})
        self.consistency_loss = CircuitConsistencyLoss(
            weight_scheme=ccfg.get("weight_scheme", "linear"),
        )
        self.supcon_loss = SupConLoss(
            temperature=float(tcfg.get("supcon_temperature", 0.07)),
        )
        self.lambda_supcon = float(tcfg.get("lambda_supcon", 0.1))

    def _build_schedulers(self):
        tcfg = self.cfg["training"]
        lcfg = tcfg["lambda_consistency"]
        self.lambda_scheduler = LambdaScheduler(
            init_val=lcfg.get("init", 0.0),
            final_val=lcfg.get("final", 1.0),
            warmup_epochs=lcfg.get("warmup_epochs", 10),
        )
        taucfg = tcfg["temperature"]
        self.tau_scheduler = TauScheduler(
            init_val=taucfg.get("init", 1.0),
            final_val=taucfg.get("final", 0.1),
            anneal_epochs=taucfg.get("anneal_epochs", 80),
        )
        self.lambda_val = self.lambda_scheduler.get(0)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(self, resume_from: str | None = None):
        start_epoch = 0
        best_val_acc = 0.0

        if resume_from is not None:
            start_epoch = self._load_checkpoint(resume_from)

        epochs = self.cfg["training"]["epochs"]
        log_interval = self.cfg["logging"].get("log_interval", 50)
        save_every = self.cfg["logging"].get("save_every", 10)

        for epoch in range(start_epoch, epochs):
            # Update auxiliary hyperparameters
            self.lambda_val = self.lambda_scheduler.get(epoch)
            tau = self.tau_scheduler.get(epoch)
            self.soft_mask.set_temperature(tau)

            # Train
            train_metrics = self._train_epoch(epoch, log_interval)

            # Validate
            val_metrics = self._val_epoch()

            self.lr_scheduler.step()

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} "
                f"task={train_metrics['task_loss']:.4f} "
                f"cons={train_metrics['cons_loss']:.4f} "
                f"supcon={train_metrics['supcon_loss']:.4f} | "
                f"val_acc={val_metrics['acc']:.3f} | "
                f"λ={self.lambda_val:.3f} τ={tau:.3f}"
            )

            # Checkpoint
            is_best = val_metrics["acc"] > best_val_acc
            if is_best:
                best_val_acc = val_metrics["acc"]
                self._save_checkpoint(epoch, val_metrics["acc"], name="best.pt")

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, val_metrics["acc"], name=f"epoch_{epoch+1}.pt")

    def _train_epoch(self, epoch: int, log_interval: int) -> dict:
        self.backbone.train()
        self.meta_encoder.train()

        total_loss = 0.0
        total_task = 0.0
        total_cons = 0.0
        total_supcon = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if self.paired:
                x1, x2, labels = batch
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.to(self.device)

                # Single forward pass for both images
                B = x1.shape[0]
                x_cat = torch.cat([x1, x2], dim=0)  # [2B, C, H, W]
                logits_cat, traj_cat = self.backbone(x_cat)

                logits1 = logits_cat[:B]
                logits2 = logits_cat[B:]
                traj1 = [h[:B] for h in traj_cat]
                traj2 = [h[B:] for h in traj_cat]

                task_loss = (
                    F.cross_entropy(logits1, labels) +
                    F.cross_entropy(logits2, labels)
                ) / 2.0

                cons_loss = self.consistency_loss(traj1, traj2)

                # Supervised contrastive loss on circuit embeddings.
                # Concatenate embeddings and labels from both views → [2B, D] / [2B].
                z1 = self.meta_encoder(traj1)
                z2 = self.meta_encoder(traj2)
                z_all = torch.cat([z1, z2], dim=0)
                labels_all = torch.cat([labels, labels], dim=0)
                supcon_loss = self.supcon_loss(z_all, labels_all)

                loss = task_loss + self.lambda_val * cons_loss + self.lambda_supcon * supcon_loss

            else:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)

                logits, traj = self.backbone(x)
                task_loss = F.cross_entropy(logits, labels)
                cons_loss = torch.zeros(1, device=self.device)
                supcon_loss = torch.zeros(1, device=self.device)
                loss = task_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.meta_encoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_cons += cons_loss.item()
            total_supcon += supcon_loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"task={task_loss.item():.4f} "
                    f"cons={cons_loss.item():.4f} "
                    f"supcon={supcon_loss.item():.4f}"
                )

        return {
            "loss": total_loss / n_batches,
            "task_loss": total_task / n_batches,
            "cons_loss": total_cons / n_batches,
            "supcon_loss": total_supcon / n_batches,
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.backbone.eval()
        self.meta_encoder.eval()

        correct = 0
        total = 0

        for batch in self.val_loader:
            if self.paired:
                x, _, labels = batch
            else:
                x, labels = batch

            x = x.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.backbone(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {"acc": correct / total}

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, val_acc: float, name: str):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "val_acc": val_acc,
                "backbone_state": self.backbone.state_dict(),
                "meta_encoder_state": self.meta_encoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone_state"])
        self.meta_encoder.load_state_dict(ckpt["meta_encoder_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resumed from {path} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.3f})")
        return ckpt["epoch"] + 1
