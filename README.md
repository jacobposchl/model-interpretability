# Phase 1: Meta-Encoder Validation

A self-supervised framework for learning interpretable representations of neural network computational structure. The meta-encoder reads a frozen backbone's activation trajectories and maps them into a **circuit space** where geometric proximity reflects shared internal computation.

## Core Idea

Neural networks reuse recurring computational pathways — stable patterns of activation across contiguous layers. We call these **circuits**. The meta-encoder learns per-layer representations `z_1, ..., z_L` such that:

- Inputs processed similarly by the backbone at a given layer are close in z-space at that layer
- The layer-by-layer structure of z-space reveals *where* in the network similarity occurs
- Discovered circuits span identifiable, contiguous depth ranges

## Architecture

```
Input x
    |
[Frozen Backbone (ResNet18)]
    |
h_l(x) --> GAP --> L2-normalize --> detach
    |
[Per-layer projectors: Linear -> GELU -> LayerNorm]
    |
p_1, ..., p_L
    |
[RoPE Transformer Encoder]
    |
z_1, ..., z_L  (L2-normalized per-layer circuit representations)
```

## Training Objective

```
L = L_info + lambda * L_geometry
```

- **L_info** (fidelity): MLP predicts per-layer cosine similarity from `z_l^a * z_l^b`
- **L_geometry** (structure): soft contrastive loss using alignment profile as target distribution

No class labels are used in training. The signal comes entirely from the backbone's internal alignment profiles.

## Circuit Discovery

Post-training, circuits are discovered via **span-centric clustering**:

1. Enumerate all `L(L+1)/2` contiguous spans `[l_start, l_end]`
2. For each span: extract within-span profile sub-vector, apply temperature sharpening, cluster with HDBSCAN
3. A pair can belong to circuits at multiple spans (multi-circuit membership)
4. Canonical circuits = clusters with >1% of all pairs

## Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Profile Reconstruction R² | >= 0.7 |
| 2 | Geometric Consistency (Spearman ρ) | > 0.5/layer, > 0.65 mean |
| 3 | Within-Span Similarity Elevation | cluster mean > pop mean + 1σ |
| 4 | Circuit Diversity | >= 60% layer coverage |
| 5 | Class Purity Distribution | bimodal (agnostic + specific) |

## Repository Structure

```
models/
  backbone.py          # Frozen backbone with configurable pooling
  meta_encoder.py      # RoPE transformer, per-layer projectors, profile regressor
losses/
  info_loss.py         # L_info: profile reconstruction fidelity
  geometry_loss.py     # L_geometry: soft contrastive with profile targets
training/
  unified_trainer.py   # Phase 1 training loop
  schedulers.py        # Lambda warmup scheduler
evaluation/
  metrics.py           # 5 success criteria functions
  discovery.py         # Span-centric circuit discovery pipeline
  circuit_analysis.py  # Data collection and profile computation
  circuit_viz.py       # UMAP, heatmaps, circuit visualizations
data/
  cifar.py             # CIFAR-10 data loading
configs/
  phase1.yaml          # Main training config
  ablations/           # Info-only, geometry-only, pooling ablations
scripts/
  train.py             # CLI training entry point
  evaluate.py          # CLI evaluation entry point
notebooks/
  experiments/         # Experiment notebooks (1-7)
documents/
  newest_iteration.md  # Detailed technical specification
tests/
  test_meta_encoder.py # RoPE, MetaEncoder, ProfileRegressor tests
  test_losses.py       # InfoLoss, GeometryLoss tests
  test_discovery.py    # Span enumeration, sharpening, discovery tests
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train the meta-encoder
python scripts/train.py --config configs/phase1.yaml

# Evaluate with success criteria
python scripts/evaluate.py --config configs/phase1.yaml \
    --checkpoint experiments/phase1/best.pt

# Run circuit discovery
python scripts/evaluate.py --config configs/phase1.yaml \
    --checkpoint experiments/phase1/best.pt --discover

# Generate visualizations
python scripts/evaluate.py --config configs/phase1.yaml \
    --checkpoint experiments/phase1/best.pt --viz
```

## Validation Experiments

1. **Profile Reconstruction Fidelity** — R² of MLP regressor + L_info/L_geometry ablations
2. **Geometric Consistency** — Per-layer Spearman ρ + UMAP visualization
3. **Circuit Discovery & Span Validation** — Span-centric clustering + multi-circuit membership
4. **Pooling Strategy Ablation** — GAP vs max vs top-k
5. **Temperature Sensitivity** — τ_geometry × τ_discovery grid search
6. **Transfer Across Backbone Depth** — ResNet18/34/50
7. **Dataset Generalization** — CIFAR-100, STL-10
