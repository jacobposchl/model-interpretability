# Circuit Trajectory Latent Space (CTLS)
### Training Neural Networks to Be Interpretable By Design

---

## Overview

Most interpretability research asks: *how do we understand what a trained model learned?* CTLS asks something fundamentally different: *what if the model was trained to be understandable in the first place?*

Instead of treating interpretability as post-hoc analysis, CTLS enforces it as a training-time structural constraint. During training, the full activation trajectory of each input — the population-level firing patterns across every intermediate layer — is embedded into a shared latent space. A consistency loss then enforces that semantically similar inputs produce similar trajectories in that space, while semantically different inputs remain well-separated.

The result is a model whose internal reasoning pathways are not discovered after the fact, but actively shaped during training to be consistent, structured, and human-legible.

---

## Core Mechanism

**Circuit definition.** A circuit is the full activation trajectory of an input through the model:

```
T(x) = (h₁(x), h₂(x), ..., h_L(x))
```

where `h_l(x)` is the vector of activations at layer `l`. This captures *how the model reasoned*, not just what it concluded.

**Meta-encoder.** A lightweight encoder `E` compresses the trajectory into a circuit embedding:

```
z = E(T(x)) = E(h₁(x), h₂(x), ..., h_L(x))
```

The meta-encoder is a small 2–3 layer MLP or lightweight transformer over the layer sequence. Its job is to produce a space where geometric distances are semantically meaningful.

**Consistency loss.** Applied to pairs of same-category inputs, penalized more heavily at later layers:

```
ℒ_cons(x₁, x₂) = Σ_l  w_l · D(h_l(x₁), h_l(x₂))
```

where `w_l` increases with layer depth (early layers encode surface features that legitimately vary; later layers encode semantic content that should be consistent).

**Total training objective:**

```
ℒ_total = ℒ_task + λ · ℒ_cons
```

**Soft masking.** Activations use magnitude-weighted soft gates rather than binary masks:

```
S_i = σ(a_i / τ) · a_i
```

Temperature `τ` is annealed over training — starting fluid, hardening into discrete, interpretable circuits.

---

## Why Circuit Space Differs From Output Space

This is the central theoretical claim. The output embedding captures *what the model concluded*. The circuit latent space captures *how it reasoned there*.

Example: a clear photograph of a dog and a heavily occluded photograph of a dog may produce nearly identical output embeddings (both classified correctly as "dog"). But their activation trajectories differ substantially — different early-layer feature processing, different attentional routing. The circuit latent space reflects this. The output space does not.

The circuit latent space contains strictly more information than the output space. And because it is shaped by the consistency loss during training, that structure is faithful by construction — not a post-hoc approximation.

---

## Novelty

The specific combination absent from existing literature:

| What | How |
|------|-----|
| Unit of analysis | Full multi-layer activation trajectory (not single-layer, not output) |
| Training signal | Semantic consistency between different same-category inputs |
| Architecture | Joint latent space over trajectories via lightweight meta-encoder |
| Contrast with prior work | Perturbation-based methods (ACT, MFNs) use the same input under noise; CTLS uses different inputs of the same class |

See [Novelty Comparison Table](#novelty-comparison) below.

---

## Theoretical Advantages

**Representational space.** With N neurons, neuron-level analysis has N dimensions. Population-level analysis has ~2^N possible activation patterns — an exponentially larger space. The model has vastly more room to encode distinct circuits per category without superposition pressure.

**Connection to generalization.** Research on grokking in Mixture-of-Experts models shows that routing pathways spontaneously become more consistent across same-category inputs at the generalization transition. The consistency loss may accelerate this transition rather than fight it.

**Faithfulness by construction.** There is no gap between the explanation and the computation. The model is literally optimized to route same-category inputs through similar pathways.

---

## Repo Structure

```
model_interpretability/
│
├── configs/                        # YAML configs — one per experiment stage
│   ├── baseline.yaml               # Stage 1: no consistency loss
│   ├── ctls.yaml                   # Stage 2: full CTLS objective
│   └── ablations/
│       ├── uniform_weighting.yaml  # Stage 4: uniform vs depth-weighted loss
│       └── no_soft_mask.yaml       # Ablation: binary masks instead of soft
│
├── models/
│   ├── backbone.py                 # ViT/ResNet wrapped with forward hooks for trajectory capture
│   ├── meta_encoder.py             # MLP/small transformer: trajectory → circuit embedding z
│   └── soft_mask.py                # Magnitude-weighted gating + temperature annealing
│
├── losses/
│   ├── consistency.py              # Depth-weighted ℒ_cons (core CTLS loss)
│   └── contrastive.py              # Contrastive objective for Stage 1 meta-encoder pretraining
│
├── data/
│   └── cifar.py                    # CIFAR-10 loader with paired same-class sampling
│
├── training/
│   ├── trainer.py                  # Training loop, λ annealing, checkpointing
│   └── schedulers.py               # LR and temperature (τ) schedules
│
├── evaluation/
│   ├── circuit_viz.py              # UMAP/t-SNE: circuit space vs output embedding space
│   ├── monosemanticity.py          # SAE-based monosemanticity scoring (Stage 5)
│   └── embedding_compare.py        # Stage 3: distance analysis — output vs circuit space
│
├── notebooks/                      # One notebook per experiment stage
│   ├── stage1_baseline.ipynb
│   ├── stage2_ctls.ipynb
│   ├── stage3_embedding_compare.ipynb
│   ├── stage4_ablation.ipynb
│   └── stage5_interpretability.ipynb
│
├── scripts/
│   ├── train.py                    # CLI entry point
│   └── evaluate.py
│
├── tests/
│   ├── test_losses.py
│   └── test_meta_encoder.py
│
└── experiments/                    # gitignored — checkpoints, logs, UMAP outputs
```

---

## Experiment Stages

### Stage 1 — Baseline Circuit Latent Space
Train a ViT or ResNet on CIFAR-10 with no consistency loss. Attach the meta-encoder and train it with a standard contrastive objective. Visualize circuit latent space via UMAP.

**Goal:** Establish how much semantic structure exists in the circuit space of a normally-trained model.

### Stage 2 — Add the Consistency Loss
Retrain with the full CTLS objective (`ℒ_task + λ · ℒ_cons`). Compare UMAP against Stage 1.

**Goal:** Do clusters tighten? Does task performance change? This is the core empirical test.

### Stage 3 — Verify the Output Embedding Distinction
Take pairs of inputs that produce similar output embeddings but differ in surface properties (e.g., clear vs. occluded same-category images). Measure pairwise distances in output embedding space vs. circuit latent space.

**Goal:** Demonstrate that circuit space contains information that output space discards.

### Stage 4 — Depth-Weighting Ablation
Compare uniform layer weighting (`w_l = const`) vs. depth-weighted (`w_l` increasing with `l`).

**Goal:** Validate the depth-weighting design decision. Without this ablation, the loss schedule is underspecified.

### Stage 5 — Interpretability Evaluation
Run SAE analysis on CTLS-trained vs. baseline models. Measure monosemanticity scores, feature purity, and circuit reuse rates across categories.

**Goal:** Bridge to the existing mechanistic interpretability literature and provide a quantitative interpretability comparison.

---

## Novelty Comparison

| Method | Mechanism | What It Misses |
|--------|-----------|----------------|
| Sparse Autoencoders (SAEs) | Post-hoc single-layer decomposition | Training-time objective; multi-layer trajectories; semantic consistency |
| Monosemantic Feature Neurons (MFNs) | Stability loss on bottleneck under noise | Cross-input semantic consistency; depth-weighted regularization |
| MonoLoss | Differentiable monosemanticity score per neuron | Circuit-level routing; population dynamics; trajectory embedding |
| Activation Consistency Training (ACT) | Residual stream consistency under prompt perturbation | Interpretability goal; semantic grouping; circuit extraction |
| Brain-Inspired Modular Training (BIMT) | Weight-level penalty on connection length | Activation trajectories; training-time consistency; latent circuit space |
| CKA / Probing Classifiers | Post-hoc representation similarity analysis | Training-time objective; trajectory embedding; any feedback to model |
| **CTLS (this project)** | Joint latent space over full multi-layer trajectories with depth-weighted semantic consistency loss | — |

---

## Technical Challenges

**Memory overhead.** Storing full activation trajectories is 3–5x the memory cost of a standard forward pass. Primary mitigation: gradient checkpointing (recompute activations during backward pass rather than storing them).

**Feature drift.** Intermediate representations shift substantially during early training. The meta-encoder must be trained jointly with the backbone. Mitigation: warm-up period with λ annealed in gradually.

**Consistency-capacity tension.** Forcing same-category circuits to converge too aggressively in early layers degrades handling of intra-class input diversity. Layer weights `w_l` must be set low for early layers and high for later layers.

**Evaluating interpretability.** Concretely measured via: (1) SAE monosemanticity scores on CTLS vs. baseline, (2) UMAP cluster separability of circuit latent space vs. output embedding space.

---

## Setup

```bash
pip install -r requirements.txt
```

**Train (baseline, no consistency loss):**
```bash
python scripts/train.py --config configs/baseline.yaml
```

**Train (full CTLS objective):**
```bash
python scripts/train.py --config configs/ctls.yaml
```

**Evaluate:**
```bash
python scripts/evaluate.py --config configs/ctls.yaml --checkpoint experiments/ctls/best.pt
```

---

## Compute Requirements

- **Stages 1–4:** Single GPU (RTX 4090 or A100). 1–3 days per stage. Google Colab Pro+ sufficient for exploration.
- **Stage 5 (SAE analysis at scale):** Small multi-GPU setup preferred. Second-paper territory.

---

## Confidence Assessment

| Dimension | Assessment | Confidence |
|-----------|-----------|------------|
| Novelty of full combination | No canonical paper implements class-based trajectory consistency as training objective | 8/10 |
| Theoretical soundness | Population space argument is well-grounded; depth-weighting has empirical support | 8/10 |
| Technical feasibility | Implementable with existing tools; memory overhead manageable at proof-of-concept scale | 7/10 |
| Superposition risk | Significantly reduced vs. neuron-level methods due to exponential population space | 7/10 |
| Output vs. circuit space distinction | Theoretically clear; empirically needs verification | 6/10 |
| **Overall** | Strong enough to build; clear enough to test; novel enough to publish if results hold | **7.5/10** |

---

## Biological Motivation

The inspiration for population-level circuit analysis comes from systems neuroscience. The brain does not encode information in single neurons — individual neurons are noisy, unreliable, and often respond to multiple unrelated stimuli. Information is encoded in population dynamics: the specific pattern of co-activation across many neurons simultaneously.

This is why neuroscientists use PCA, UMAP, and trajectory analysis on population recordings rather than analyzing individual neurons in isolation. The representational space of population patterns is exponentially larger than the space of individual neuron activations — concepts are trajectories through neural population space, not locations in single-neuron space.

CTLS borrows this insight directly. The model does not need to cheat by overlapping concepts in single neurons (superposition) when it has vast population space to work with.
