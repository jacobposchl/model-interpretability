"""
Phase 1 Trainer: Meta-Encoder Validation.

Trains a meta-encoder to learn circuit-space representations from a frozen
backbone's activation trajectories. The backbone is never modified.

    L_total = L_info + lambda * L_geometry

where:
  L_info:     Profile reconstruction fidelity (MLP on z_l^a * z_l^b vs s_l)
  L_geometry: Soft contrastive geometry (profile-weighted cross-entropy in z-space)

All pairs are formed within-batch from standard CIFAR-10 batches. No class-label
pairing is needed — the training signal comes entirely from alignment profiles.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.backbone import FrozenBackbone
from models.meta_encoder import MetaEncoder, ProfileRegressor
from losses.info_loss import InfoLoss
from losses.geometry_loss import GeometryLoss
from data.cifar import get_standard_loaders
from training.schedulers import LambdaScheduler


class Phase1Trainer:
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
        mcfg = self.cfg["model"]

        self.backbone = FrozenBackbone(
            arch=mcfg["arch"],
            num_classes=mcfg.get("num_classes", 10),
            pretrained=mcfg.get("pretrained", True),
            pool_mode=mcfg.get("pool_mode", "gap"),
        ).to(self.device)

        ecfg = mcfg["meta_encoder"]
        self.meta_encoder = MetaEncoder(
            layer_dims=self.backbone.layer_dims,
            projection_dim=ecfg.get("projection_dim", 128),
            n_heads=ecfg.get("n_heads", 4),
            n_transformer_layers=ecfg.get("n_transformer_layers", 2),
            dropout=ecfg.get("dropout", 0.0),
        ).to(self.device)

        rcfg = mcfg.get("regressor", {})
        self.regressor = ProfileRegressor(
            input_dim=ecfg.get("projection_dim", 128),
            hidden_dim=rcfg.get("hidden_dim", 64),
        ).to(self.device)

    def _build_data(self):
        dcfg = self.cfg["data"]
        self.train_loader, self.val_loader = get_standard_loaders(
            data_dir=dcfg.get("data_dir", "data/cifar10"),
            batch_size=dcfg.get("batch_size", 256),
            num_workers=dcfg.get("num_workers", 4),
            augment=dcfg.get("augment", True),
            download=True,
        )

    def _build_optimizers(self):
        tcfg = self.cfg["training"]
        lr = float(tcfg.get("lr", 1e-3))
        # Only meta-encoder and regressor are trained; backbone is frozen
        params = list(self.meta_encoder.parameters()) + list(self.regressor.parameters())
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
        self.info_loss = InfoLoss(regressor=self.regressor)
        self.geometry_loss = GeometryLoss(
            temperature=float(tcfg.get("geometry_temperature", 0.1))
        )
        self.info_loss_weight = float(tcfg.get("info_loss_weight", 1.0))

    def _build_schedulers(self):
        tcfg = self.cfg["training"]
        lcfg = tcfg.get("lambda_geometry", {})
        self.lambda_scheduler = LambdaScheduler(
            init_val=lcfg.get("init", 0.0),
            final_val=lcfg.get("final", 1.0),
            warmup_epochs=lcfg.get("warmup_epochs", 10),
        )
        self.lambda_val = self.lambda_scheduler.get(0)

    # ------------------------------------------------------------------ #
    # Profile computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_profiles(trajectory: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute pairwise alignment profiles from L2-normalized trajectory.

        Args:
            trajectory: list of L tensors, each [B, D_l], L2-normalized

        Returns:
            profiles: [B, B, L] pairwise per-layer cosine similarities
        """
        L = len(trajectory)
        B = trajectory[0].shape[0]
        device = trajectory[0].device

        profiles = torch.empty(B, B, L, device=device)
        for l in range(L):
            # h_l is already L2-normalized, so dot product = cosine sim
            profiles[:, :, l] = trajectory[l] @ trajectory[l].t()

        return profiles

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(self, resume_from: str | None = None):
        start_epoch = 0
        best_val_r2 = -float("inf")

        if resume_from is not None:
            start_epoch = self._load_checkpoint(resume_from)

        epochs = self.cfg["training"]["epochs"]
        log_interval = self.cfg["logging"].get("log_interval", 50)
        save_every = self.cfg["logging"].get("save_every", 10)

        for epoch in range(start_epoch, epochs):
            self.lambda_val = self.lambda_scheduler.get(epoch)

            train_metrics = self._train_epoch(epoch, log_interval)
            val_metrics = self._val_epoch()
            self.lr_scheduler.step()

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} "
                f"info={train_metrics['info_loss']:.4f} "
                f"geom={train_metrics['geometry_loss']:.4f} | "
                f"val_R2={val_metrics['r2']:.3f} "
                f"val_rho={val_metrics['mean_rho']:.3f} | "
                f"lambda={self.lambda_val:.3f}"
            )

            is_best = val_metrics["r2"] > best_val_r2
            if is_best:
                best_val_r2 = val_metrics["r2"]
                self._save_checkpoint(epoch, val_metrics, name="best.pt")

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, val_metrics, name=f"epoch_{epoch+1}.pt")

    def _train_epoch(self, epoch: int, log_interval: int) -> dict:
        self.meta_encoder.train()
        self.regressor.train()

        total_loss = 0.0
        total_info = 0.0
        total_geom = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            images = images.to(self.device)
            B = images.shape[0]

            # Forward through frozen backbone (no grad, already detached)
            trajectory = self.backbone(images)

            # Compute ground-truth alignment profiles [B, B, L]
            profiles = self.compute_profiles(trajectory)

            # Forward through meta-encoder
            z_list = self.meta_encoder(trajectory)  # list of L tensors [B, d]

            # --- L_info ---
            # Form all unique pairs using upper-triangle indices
            idx_a, idx_b = torch.triu_indices(B, B, offset=1, device=self.device)

            # Gather per-layer z-vectors for each pair
            z_pairs_a = [z_l[idx_a] for z_l in z_list]  # list of L x [N_pairs, d]
            z_pairs_b = [z_l[idx_b] for z_l in z_list]

            # Gather true similarities for pairs: [N_pairs, L]
            true_sims_pairs = profiles[idx_a, idx_b, :]  # [N_pairs, L]

            info_loss = self.info_loss(z_pairs_a, z_pairs_b, true_sims_pairs)

            # --- L_geometry ---
            geometry_loss = self.geometry_loss(z_list, profiles)

            # --- Total ---
            loss = self.info_loss_weight * info_loss + self.lambda_val * geometry_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.meta_encoder.parameters()) + list(self.regressor.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_info += info_loss.item()
            total_geom += geometry_loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"info={info_loss.item():.4f} "
                    f"geom={geometry_loss.item():.4f}"
                )

        return {
            "loss": total_loss / max(n_batches, 1),
            "info_loss": total_info / max(n_batches, 1),
            "geometry_loss": total_geom / max(n_batches, 1),
        }

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.meta_encoder.eval()
        self.regressor.eval()

        all_predicted = []
        all_true = []
        per_layer_z_sims = []
        per_layer_true_sims = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            B = images.shape[0]
            if B < 2:
                continue

            trajectory = self.backbone(images)
            profiles = self.compute_profiles(trajectory)
            z_list = self.meta_encoder(trajectory)

            # Profile reconstruction (for R^2)
            idx_a, idx_b = torch.triu_indices(B, B, offset=1, device=self.device)
            z_pairs_a = [z_l[idx_a] for z_l in z_list]
            z_pairs_b = [z_l[idx_b] for z_l in z_list]
            true_sims = profiles[idx_a, idx_b, :]  # [N_pairs, L]

            L = len(z_list)
            batch_pred = []
            for l in range(L):
                z_product = z_pairs_a[l] * z_pairs_b[l]
                pred_l = self.regressor(z_product)  # [N_pairs]
                batch_pred.append(pred_l)

            predicted = torch.stack(batch_pred, dim=1)  # [N_pairs, L]
            all_predicted.append(predicted.cpu())
            all_true.append(true_sims.cpu())

            # Geometric consistency (for Spearman rho)
            for l in range(L):
                z_sim_l = (z_list[l] @ z_list[l].t())  # [B, B]
                true_sim_l = profiles[:, :, l]
                # Upper triangle only
                z_flat = z_sim_l[idx_a, idx_b].cpu().numpy()
                t_flat = true_sim_l[idx_a, idx_b].cpu().numpy()
                per_layer_z_sims.append((l, z_flat))
                per_layer_true_sims.append((l, t_flat))

        # Compute R^2
        all_predicted = torch.cat(all_predicted, dim=0)
        all_true = torch.cat(all_true, dim=0)
        ss_res = ((all_predicted - all_true) ** 2).sum().item()
        ss_tot = ((all_true - all_true.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        # Compute per-layer Spearman rho
        from collections import defaultdict
        import numpy as np
        layer_z = defaultdict(list)
        layer_t = defaultdict(list)
        for l, z_flat in per_layer_z_sims:
            layer_z[l].append(z_flat)
        for l, t_flat in per_layer_true_sims:
            layer_t[l].append(t_flat)

        per_layer_rho = []
        for l in sorted(layer_z.keys()):
            z_all = np.concatenate(layer_z[l])
            t_all = np.concatenate(layer_t[l])
            rho, _ = spearmanr(z_all, t_all)
            per_layer_rho.append(rho if not np.isnan(rho) else 0.0)

        mean_rho = float(np.mean(per_layer_rho)) if per_layer_rho else 0.0

        return {
            "r2": r2,
            "mean_rho": mean_rho,
            "per_layer_rho": per_layer_rho,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, val_metrics: dict, name: str):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "val_metrics": val_metrics,
                "meta_encoder_state": self.meta_encoder.state_dict(),
                "regressor_state": self.regressor.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.meta_encoder.load_state_dict(ckpt["meta_encoder_state"])
        self.regressor.load_state_dict(ckpt["regressor_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        metrics = ckpt.get("val_metrics", {})
        print(
            f"Resumed from {path} (epoch {ckpt['epoch']}, "
            f"R2={metrics.get('r2', 'N/A')}, rho={metrics.get('mean_rho', 'N/A')})"
        )
        return ckpt["epoch"] + 1
