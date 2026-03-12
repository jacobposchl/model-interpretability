"""
CLI entry point for evaluation.

Loads a trained checkpoint and runs one or more evaluation stages.

Examples:
    # Stage 3: compare circuit vs output embedding spaces
    python scripts/evaluate.py --config configs/ctls.yaml \\
        --checkpoint experiments/ctls/best.pt --stage 3

    # Stage 5: monosemanticity scoring (requires baseline checkpoint too)
    python scripts/evaluate.py --config configs/ctls.yaml \\
        --checkpoint experiments/ctls/best.pt \\
        --baseline-checkpoint experiments/baseline/best.pt --stage 5

    # Full UMAP visualisation (no stage flag = run viz)
    python scripts/evaluate.py --config configs/ctls.yaml \\
        --checkpoint experiments/ctls/best.pt --viz
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path

from models.soft_mask import SoftMask
from models.backbone import CTLSBackbone
from models.meta_encoder import MetaEncoder
from data.cifar import get_standard_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CTLS model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="Baseline checkpoint for Stage 5 comparison")
    parser.add_argument("--stage", type=int, default=None,
                        choices=[3, 4, 5],
                        help="Evaluation stage to run (3, 4, or 5)")
    parser.add_argument("--viz", action="store_true",
                        help="Run UMAP/t-SNE visualisation")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save figures (defaults to checkpoint dir)")
    return parser.parse_args()


def load_model(config: dict, checkpoint_path: str, device: torch.device):
    mcfg = config["model"]
    ecfg = mcfg["meta_encoder"]
    tcfg = config["training"]

    soft_mask = SoftMask(init_temperature=tcfg["temperature"]["init"]).to(device)
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
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.3f})")

    return backbone, meta_encoder


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = config["data"]

    backbone, meta_encoder = load_model(config, args.checkpoint, device)
    backbone.eval()
    meta_encoder.eval()

    _, val_loader = get_standard_loaders(
        data_dir=dcfg["data_dir"],
        batch_size=dcfg["batch_size"],
        num_workers=dcfg.get("num_workers", 4),
        augment=False,
    )

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # UMAP / t-SNE visualisation
    # ------------------------------------------------------------------ #
    if args.viz:
        from evaluation.circuit_viz import CircuitVisualizer
        viz = CircuitVisualizer(backbone, meta_encoder, val_loader, device)

        print("Computing UMAP...")
        fig = viz.plot_umap(
            title=config["experiment"]["name"],
            compare_output=True,
        )
        path = output_dir / "umap_circuit_vs_output.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

        scores = viz.cluster_separation_score()
        print(f"Silhouette — circuit: {scores['silhouette_circuit']:.4f}, "
              f"output: {scores['silhouette_output']:.4f}, "
              f"delta: {scores['delta']:.4f}")

    # ------------------------------------------------------------------ #
    # Stage 3: circuit vs output distance comparison
    # ------------------------------------------------------------------ #
    if args.stage == 3:
        from evaluation.embedding_compare import EmbeddingComparator
        comp = EmbeddingComparator(backbone, meta_encoder, device)

        print("Stage 3: clean vs degraded distance comparison...")
        results = comp.compare_clean_vs_degraded(val_loader)
        print(f"  Output dist mean:  {results['output_dist_mean']:.4f}")
        print(f"  Circuit dist mean: {results['circuit_dist_mean']:.4f}")
        print(f"  Ratio (circuit/output): {results['ratio_circuit_over_output']:.3f}")

        fig = comp.plot_distance_comparison(val_loader)
        path = output_dir / "stage3_distance_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    # ------------------------------------------------------------------ #
    # Stage 5: monosemanticity scoring
    # ------------------------------------------------------------------ #
    if args.stage == 5:
        from evaluation.monosemanticity import MonosemanticityScorer

        scorer = MonosemanticityScorer(backbone, val_loader, device)

        if args.baseline_checkpoint:
            baseline_backbone, _ = load_model(config, args.baseline_checkpoint, device)
            baseline_backbone.eval()
            comparison = scorer.compare_with_baseline(baseline_backbone)
            print("\nLayer-by-layer monosemanticity comparison (CTLS vs baseline):")
            for row in comparison["layer_results"]:
                print(
                    f"  Layer {row['layer_idx']+1:2d} | "
                    f"mono: {row['base_mono']:.3f} → {row['ctls_mono']:.3f} "
                    f"(Δ={row['delta_mono']:+.3f}) | "
                    f"reuse: {row['base_reuse']:.3f} → {row['ctls_reuse']:.3f} "
                    f"(Δ={row['delta_reuse']:+.3f})"
                )
        else:
            results = scorer.score_all_layers()
            print("\nMonosemanticity scores by layer:")
            for r in results:
                print(
                    f"  Layer {r['layer_idx']+1:2d} | "
                    f"mono={r['monosemanticity_score']:.3f} "
                    f"entropy={r['mean_feature_entropy']:.3f} "
                    f"reuse={r['circuit_reuse_rate']:.3f}"
                )


if __name__ == "__main__":
    main()
