# Trainable Circuits
### Circuit Trajectory Latent Space (CTLS) — Training Neural Networks to Be Interpretable By Design

---

## What This Project Is

Most interpretability research asks *how do we understand what a trained model learned?* This project asks something different: *what if the model was trained to be understandable in the first place?*

CTLS enforces interpretability as a training-time structural constraint. During training, the full multi-layer activation trajectory of each input is embedded into a shared latent space via a lightweight meta-encoder. A contrastive consistency loss (InfoNCE) then forces semantically similar inputs to produce similar trajectories in that space, while semantically different inputs remain separated. The result is a model whose internal reasoning pathways are actively shaped during training to be consistent and semantically organized — not discovered after the fact.

For the full scientific context, architecture details, and motivation: [documents/research_context.md](documents/research_context.md)

For all experimental results: [documents/results.md](documents/results.md)

For planned experiments and future directions: [documents/next_steps.md](documents/next_steps.md)

---

## Status

- **Steps 1–3 complete:** Unified objective implemented and validated. Trajectory cosine similarity tracks per-layer activation similarity at Spearman ρ = 0.797/0.781 (95% CI [0.775, 0.816] / [0.756, 0.801]) against a continuous per-layer ground truth.
- **Key finding:** Class-discriminative structure concentrates in layers 7–8 (gap 0.53 at layer 8 vs 0.015 at layer 1). z is dominated by those layers (per-layer ρ = 0.91 at layer 8). z ≈ h₈ in discriminative power, indicating class-level collapse from class-label positive pairs.
- **Step 5 in progress:** Positive pair redefinition — replacing class-label pairs with trajectory-similarity-derived pairs to break class-level collapse and force multi-layer circuit organization.
- **Steps 4, 6 pending:** Activation extraction ablation, SSL extension.

---

## Key Results

| Metric | Baseline | CTLS (unified) |
|--------|---------|----------------|
| Val accuracy | 93.53% | 94.21% |
| Circuit silhouette | 0.149 | 0.819 |
| Output silhouette | 0.807 | 0.824 |
| Intraclass ρ (mean) | 0.295 | 0.714 |
| Noise ratio (σ=0.3) | 0.295 | 0.818 |
| Proxy ρ vs. per-layer traj sim | — | 0.797 / 0.781 |
| Proxy ρ 95% CI | — | [0.775, 0.816] / [0.756, 0.801] |

---

## Repo Structure

```
trainable-circuits/
│
├── documents/
│   ├── research_context.md         # Project idea, architecture, theory, and scientific context
│   ├── results.md                  # All experimental results (Stages 1–5 + validation Steps 1–3)
│   └── next_steps.md               # Planned experiments and future directions (with status tracking)
│
├── configs/
│   ├── unified_a.yaml              # Current — Option A: fixed depth-ramp meta-encoder
│   ├── unified_b.yaml              # Current — Option B: transformer CLS meta-encoder
│   ├── baseline.yaml               # No consistency loss (reference baseline)
│   ├── ablations/
│   │   ├── uniform_weighting.yaml  # Uniform vs depth-weighted ablation
│   │   └── no_soft_mask.yaml       # Binary masks instead of soft
│   └── ssl/
│       ├── ctls_ssl_v2.yaml        # CTLS-SSL (Step 6, config template)
│       ├── dino.yaml               # DINO baseline for SSL comparison
│       └── simclr.yaml             # SimCLR baseline for SSL comparison
│
├── models/
│   ├── backbone.py                 # ResNet/ViT with forward hooks for trajectory capture
│   ├── meta_encoder.py             # weighted_sum and transformer_cls variants
│   ├── momentum_encoder.py         # Momentum encoder + embedding bank (for SSL, Step 6)
│   └── soft_mask.py                # Magnitude-weighted gating + temperature annealing
│
├── losses/
│   ├── simclr.py                   # NTXentLoss — InfoNCE for the unified objective
│   └── dino_loss.py                # DINO loss + projection head (for SSL, Step 6)
│
├── data/
│   ├── cifar.py                    # CIFAR-10 with paired same-class sampling
│   └── ssl.py                      # SSL augmentation pipeline and CIFAR-100 transfer datasets
│
├── training/
│   ├── unified_trainer.py          # Trainer for the unified CTLS objective
│   └── schedulers.py               # λ warmup and soft mask temperature schedules
│
├── evaluation/
│   ├── activation_patching.py      # Causal circuit similarity via activation patching
│   ├── circuit_analysis.py         # Circuit structure analysis utilities
│   ├── circuit_viz.py              # UMAP/t-SNE: circuit space vs output embedding space
│   ├── embedding_compare.py        # Distance analysis — output vs circuit space
│   ├── fewshot.py                  # Few-shot evaluation for SSL experiments
│   └── monosemanticity.py          # SAE-based monosemanticity scoring
│
├── notebooks/
│   └── validation/
│       └── nb01_validation_experiment.ipynb    # Steps 1–3: unified objective + proxy validation
│
├── scripts/
│   ├── train.py                    # CLI training entry point
│   └── evaluate.py                 # CLI evaluation entry point
│
├── tests/
│   ├── test_meta_encoder.py        # SoftMask, weighted_sum, transformer_cls
│   ├── test_fewshot.py             # EpisodeSampler, FewShotEvaluator, EmbeddingBank
│   └── test_ssl_losses.py          # NTXentLoss, DINOLoss
│
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

**Train baseline (no consistency loss):**
```bash
python scripts/train.py --config configs/baseline.yaml
```

**Train CTLS — Option A (fixed depth-ramp meta-encoder):**
```bash
python scripts/train.py --config configs/unified_a.yaml
```

**Train CTLS — Option B (transformer CLS meta-encoder):**
```bash
python scripts/train.py --config configs/unified_b.yaml
```

**Evaluate a checkpoint:**
```bash
python scripts/evaluate.py --config configs/unified_b.yaml --checkpoint experiments/unified_b/best.pt
```

The active experiment notebook is [notebooks/validation/nb01_validation_experiment.ipynb](notebooks/validation/nb01_validation_experiment.ipynb).

---

## Architecture in Brief

The core pipeline: per-layer projectors map each activation vector to a common dimension `d`, a depth-aware meta-encoder combines them into a single circuit embedding `z`, and InfoNCE pulls same-class `z` vectors together while pushing other-class vectors apart.

```
h_l(x) ──► Linear_l + LayerNorm + GELU ──► p_l ∈ R^d
                                              │
                           [p₁, ..., p_L] ──► Meta-Encoder ──► z ∈ R^64 (L2-norm)
                                                                  │
                                         L_total = L_task + λ · InfoNCE(z₁, z₂)
```

**Option A (`weighted_sum`):** `z = Σ_l w_l · p_l` with a fixed linear depth ramp. Interpretable, auditable distances.

**Option B (`transformer_cls`):** 2-layer transformer with CLS pooling and sinusoidal depth-encoding positional embeddings. Learns which layers matter per input. Achieves ρ = 0.743 vs 0.717 for Option A on proxy validation.

Both configs are in [configs/unified_a.yaml](configs/unified_a.yaml) and [configs/unified_b.yaml](configs/unified_b.yaml).
