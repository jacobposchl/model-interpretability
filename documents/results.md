# CTLS Experimental Results
### All Experiments — Original Stages + Validation

---

## Overview

All experiments use ResNet18 on CIFAR-10. Results are organized in two phases: the original five-stage experiments that established the foundational metrics, and the validation experiment (Steps 1–3) that implemented the unified objective and validated the trajectory proxy against an independent causal ground truth.

---

## Part 1 — Original Stage Experiments (Stages 1–5)

These experiments used the original architecture and established baseline metrics. The architecture has since been refined (unified objective, single InfoNCE on depth-aware z), but these results remain valid observations and form the baseline comparisons for the validation experiment.

### Stage 1 — Baseline Characterization

**Val accuracy: 93.53%**

| Space | Silhouette |
|-------|-----------|
| Circuit | 0.1466 |
| Output | 0.7974 |

**Per-layer silhouette (ResNet18, 8 blocks):**

| Layer | Dim | Silhouette |
|-------|-----|-----------|
| 1 | 64 | −0.0809 |
| 2 | 64 | −0.0478 |
| 3 | 128 | −0.0450 |
| 4 | 128 | −0.0195 |
| 5 | 256 | +0.0179 |
| 6 | 256 | +0.0800 |
| 7 | 512 | +0.2308 |
| 8 | 512 | +0.2698 |

Layers 1–4 are genuinely anti-class-structured — within-class distances exceed between-class distances. Class identity only becomes geometrically resolvable in the final blocks. The baseline model's class structure is front-loaded into the final logit space; the activation trajectory through the network is largely unstructured. This is the gap CTLS targets.

---

### Stage 2 — Full CTLS Objective

**Checkpoint: epoch 95, val_acc = 94.21%**

| Metric | Baseline | CTLS | Delta |
|--------|---------|------|-------|
| Circuit silhouette | 0.1486 | 0.8097 | **+0.6611** |
| Output silhouette | 0.8091 | 0.8124 | +0.0033 |
| Val accuracy | 93.53% | 94.21% | **+0.68%** |

The +0.66 circuit silhouette jump is the core result. Output silhouette is unchanged (+0.003), confirming CTLS did not achieve circuit structure by collapsing the backbone. Accuracy improves slightly — the consistency pressure acts as a useful regularizer.

---

### Stage 3 — Embedding Analysis

**Intraclass rank correlations (Spearman's ρ):**

| Class | CTLS ρ | Baseline ρ | Δρ |
|-------|--------|-----------|-----|
| airplane | 0.849 | 0.198 | +0.651 |
| automobile | 0.593 | 0.210 | +0.383 |
| bird | 0.876 | 0.330 | +0.546 |
| cat | 0.904 | 0.288 | +0.616 |
| deer | 0.781 | 0.149 | +0.632 |
| dog | 0.869 | 0.357 | +0.512 |
| frog | 0.717 | 0.399 | +0.318 |
| horse | 0.675 | 0.246 | +0.429 |
| ship | 0.682 | 0.325 | +0.357 |
| truck | 0.736 | 0.450 | +0.286 |
| **Mean** | **0.768** | **0.295** | **+0.473** |

CTLS improves intraclass consistency by ~2.6× on average. Animal classes show the strongest effect (cat 0.904, bird 0.876, dog 0.869). Vehicle classes are lower, likely reflecting genuine visual similarity between automobile and truck that the circuit space captures rather than forcing artificial separation.

**Noise robustness (circuit-to-output tracking ratio):**

| Noise σ | CTLS ratio | Baseline ratio |
|---------|-----------|---------------|
| 0.05 | 0.511 | 0.417 |
| 0.10 | 0.645 | 0.585 |
| 0.20 | 0.704 | 0.364 |
| 0.30 | 0.755 | 0.295 |
| 0.50 | 0.784 | 0.273 |
| 0.80 | 0.788 | 0.299 |

The divergence at high noise is the key finding. CTLS circuit embeddings maintain informational fidelity under distribution shift; baseline embeddings cannot track what the model is computing because they were never encoding it in the first place.

---

### Stage 4 — Depth-Weighting Ablation

| Variant | Circuit sil | Output sil | Val acc |
|---------|------------|-----------|--------|
| Baseline (λ=0) | 0.1513 | 0.8040 | 93.53% |
| Uniform weights | 0.8068 | 0.8020 | 93.88% |
| Depth-weighted | **0.8295** | **0.8309** | **94.21%** |

Depth-weighting outperforms uniform weighting on all three metrics. Notably, uniform weighting slightly degrades output silhouette vs. baseline — applying uniform consistency pressure on early layers may interfere with the backbone's development of low-level features. Depth-weighting avoids this by not penalizing early layers heavily.

**Known limitation — Layer 7 local collapse:** Both CTLS variants show a per-layer silhouette dip at layer 7. The consistency pressure creates partial representational collapse at this layer. A per-layer decorrelation or diversity penalty targeted at layer 7 is the identified fix.

---

### Stage 5 — SAE Monosemanticity

| Layer | Base mono | CTLS mono | Δmono | Base reuse | CTLS reuse | Δreuse |
|-------|----------|----------|-------|-----------|-----------|-------|
| 1 | 0.059 | 0.023 | −0.035 | 0.473 | 0.363 | −0.109 |
| 2 | 0.078 | 0.031 | −0.047 | 0.426 | 0.391 | −0.035 |
| 3 | 0.105 | 0.012 | −0.094 | 0.414 | 0.363 | −0.051 |
| 4 | 0.105 | 0.010 | −0.096 | 0.371 | 0.299 | −0.072 |
| 5 | 0.092 | 0.013 | −0.079 | 0.332 | 0.327 | −0.005 |
| 6 | 0.095 | 0.012 | −0.083 | 0.318 | 0.347 | +0.028 |
| 7 | 0.085 | 0.004 | −0.081 | 0.333 | 0.216 | −0.117 |
| 8 | 0.125 | 0.004 | −0.122 | 0.383 | 0.247 | −0.136 |
| **Avg** | | | **−0.080** | | | **−0.062** |

**SAE quality (final layer):**

| | Recon MSE | Sparsity |
|-|----------|---------|
| CTLS SAE | 0.00166 | 0.2492 |
| Baseline SAE | 0.00238 | 0.3428 |

CTLS achieves higher circuit silhouette but lower monosemanticity — the monosemanticity paradox. The resolution: CTLS encodes class identity through structured superposition (population-level patterns) rather than monosemantic individual features. The lower reconstruction MSE (0.00166 vs 0.00238) confirms CTLS representations have more regular, learnable structure even though no individual feature is class-exclusive. See `documents/research_context.md` §7 for the full interpretation.

---

## Part 2 — Validation Experiment (Steps 1–3, original)

This experiment implemented the unified objective and validated the trajectory proxy against activation-patching ground truth.

> **Note:** Steps 2–3 of this experiment used activation patching as the ground truth. That measure has a known cascade confound (patching at layer l disrupts all downstream layers simultaneously), which produced a binary CircuitSim distribution and limited interpretability. The refined validation in Part 3 supersedes Steps 2–3 with a cleaner ground truth. Step 1 (unified objective sanity check) and its results remain valid and are the reference baseline.

---

### Background: Why This Experiment Was Needed

The original CTLS implementation had two problems that prevented strong interpretability claims:

1. **Disconnected loss signals.** The depth-weighted cosine loss operated on raw activations; the supervised contrastive loss operated on meta-encoder outputs. These were architecturally separate — depth weighting never touched the meta-encoder.

2. **Circular evaluation.** The primary metric (silhouette on z) measured how well z clustered by class, but z was explicitly trained via the supervised contrastive loss to cluster by class. The metric measured convergence, not whether z encoded anything about circuit structure.

3. **Unvalidated proxy.** The entire project rested on the assumption that trajectory cosine similarity approximates causal circuit similarity. This had never been tested against independent ground truth.

---

### Step 1 — Unified Objective: Sanity Check

Both unified variants (Option A: `weighted_sum`, Option B: `transformer_cls`) were trained for 100 epochs on CIFAR-10 with λ warmed up linearly over 10 epochs, soft mask temperature cosine-annealed from 1.0 to 0.1, and InfoNCE temperature τ = 0.07.

| Metric | Option A (weighted_sum) | Option B (transformer_cls) | Original CTLS |
|--------|------------------------|---------------------------|---------------|
| Circuit silhouette | **0.8169** | **0.8194** | 0.8097 |
| Output silhouette | 0.8198 | 0.8236 | 0.8124 |
| Intraclass ρ (mean) | 0.7129 | 0.7147 | 0.7680 |
| Noise ratio (σ=0.3) | 0.8177 | 0.8007 | 0.7840 |

**Interpretation:**

The unified objective reproduces and in some cases slightly exceeds the original CTLS metrics. Removing `L_cons` and `L_supcon` and replacing them with a single InfoNCE on `z` does not degrade circuit organization — the single-signal architecture achieves equivalent results with a cleaner design.

- **Circuit silhouette (0.82 vs 0.81 original):** z-space is as class-organized as before.
- **Output silhouette unchanged (0.82):** The backbone's classification behavior is intact.
- **Intraclass ρ slightly lower (0.71 vs 0.77):** The original `L_cons` directly penalized pairwise distances in trajectory space, which directly optimized this metric. InfoNCE on z is a less direct signal. The gap is small and not concerning.
- **Option A and Option B are nearly identical on all sanity metrics.** Any difference in proxy validation is attributable to meta-encoder architecture, not backbone training quality.

---

### Step 2 — Activation Patching Ground Truth

#### Method

For each pair (x_a, x_b): for each layer l, replace x_b's raw block output at layer l with x_a's block output and measure how much the final logit distribution changes.

```
influence_l = KL( softmax(logits_b_clean) ‖ softmax(logits_b_patched_at_l) )
CircuitSim(x_a, x_b) = 1 − mean_l( influence_l / max_l(influence_l) )
```

**Important implementation detail:** The backbone's trajectory hooks store globally average-pooled activations `[B, D]`, which cannot be used as patch values — the next layer expects a spatial `[B, C, H, W]` tensor. Separate temporary capture hooks record raw pre-pooling block outputs for use as patch values, while the backbone's read-only hooks continue operating normally on the pooled versions.

1000 pairs were sampled from the CIFAR-10 validation set: 500 same-class pairs (50 per class) and 500 different-class pairs.

#### Results

**Different-class pairs: CircuitSim = 0.000 (std = 0.000)**

Every different-class pair received CircuitSim = 0. When two inputs are from different classes, patching x_a's activations into x_b's forward pass causes significant output disruption at every layer. The KL profile is flat across layers (all layers equally disrupted), so after normalizing by the max, all values equal 1.0, and `1 − mean(1.0) = 0`. Zero standard deviation confirms this holds without exception for all 500 different-class pairs. This is a strong sanity check: the patching signal cleanly discriminates between-class pairs from same-class pairs.

**Same-class pairs: CircuitSim = 0.524 (std = 0.499)**

The near-equal mean and standard deviation indicate a **bimodal distribution**. Same-class pairs split approximately 50/50 between CircuitSim ≈ 1 and CircuitSim ≈ 0.

- **CircuitSim ≈ 1 pairs:** These have a KL profile concentrated at one or two layers — patching most layers barely changes x_b's output (those layers' circuits are shared), but patching one specific layer causes a large change. These pairs genuinely share their computational circuit except at one differentiating layer.
- **CircuitSim ≈ 0 pairs:** These have a flat KL profile — every layer contributes equally to the output change when patched. Either the inputs use genuinely different circuits at every layer, or the class-relevant computation is so distributed that no single layer dominates.

**The 50/50 split is an empirical finding about ResNet18's internal structure:** approximately half of same-class image pairs share their circuit routing, and half do not, even though they produce the same predicted class. Class label and circuit identity are not the same thing.

#### Per-Layer Influence Profile

When mean KL is plotted across all 1000 pairs, the profile is flat (all layers ~21–22 KL units). This is an averaging artifact: the 500 different-class pairs have uniformly high KL at every layer, dominating the average and masking per-pair variation in same-class pairs. The per-pair variation — not the average — is where the signal lives. For CircuitSim=1 same-class pairs, the dominant layer varies from pair to pair with no consistent bias toward any particular depth.

---

### Step 3 — Proxy Validation

#### Method

For each of the 1000 pairs, compute z-space cosine similarity using the trained meta-encoders from Step 1. Compute Spearman rank correlation between z-space similarity and activation-patching CircuitSim.

#### Primary Results

| Model | Spearman ρ | p-value |
|-------|-----------|---------|
| Option A (weighted_sum, d=128) | **0.7174** | 7.32e-159 |
| Option B (transformer_cls, d=128) | **0.7428** | 4.54e-176 |

Both well exceed the validation threshold of ρ > 0.6. The p-values are effectively zero.

**The proxy is validated.** Trajectory cosine similarity, compressed through a depth-aware meta-encoder and trained with InfoNCE, tracks causal circuit similarity as measured by activation patching.

#### Within-Same-Class ρ: The Crucial Follow-up

Given the bimodal ground truth (CircuitSim ∈ {0, 1}), a concern was that the reported ρ was inflated by the trivial between-class signal — z-space similarity just needed to rank same-class pairs above different-class pairs. Computing Spearman ρ restricted to the 500 same-class pairs:

```
Within-same-class ρ = 0.7094  (p = 1.11e-77)
```

This is essentially identical to the overall ρ of 0.717. The between-class separation contributes almost nothing to the overall correlation — the reported ρ is almost entirely from within-class discrimination.

**This is the stronger result.** Among pairs of images from the same class, z-space cosine similarity correctly identifies which pairs share their circuit routing (CircuitSim ≈ 1) versus which take different internal paths despite producing the same prediction (CircuitSim ≈ 0). The model is not simply learning "cats are similar to cats." It is learning "these two cats process similarly, and those two cats do not."

---

## Part 3 — Refined Validation Experiment (nb01, rev.2)

The activation patching ground truth from Part 2 Steps 2–3 had a cascade confound: replacing x_b's activations at layer l with x_a's also corrupts all downstream layers, making it impossible to attribute the output change to layer l alone. For ResNet18 on diverse CIFAR-10 pairs, this produced a binary CircuitSim distribution (0 or 1) rather than a continuous one. The refined validation replaces patching with per-layer cosine similarity measured independently at each layer.

---

### Step 2 (revised) — Per-Layer Trajectory Similarity

**Ground truth:**

```
sim_l(x_a, x_b) = cos( h_l(x_a), h_l(x_b) )   for l = 1..8
```

No cascade. Each layer measured in isolation from a clean forward pass.

**Per-layer same/diff-class gap (mean same-class sim − mean diff-class sim):**

| Layer | Gap A (weighted_sum) | Gap B (transformer_cls) |
|-------|---------------------|------------------------|
| 1 | 0.0149 | 0.0175 |
| 2 | 0.0184 | 0.0243 |
| 3 | 0.0387 | 0.0406 |
| 4 | 0.0332 | 0.0365 |
| 5 | 0.0571 | 0.0610 |
| 6 | 0.0904 | 0.0911 |
| 7 | 0.3019 | 0.2862 |
| 8 | 0.5253 | 0.5016 |

The class-discriminative structure in ResNet18 is almost entirely concentrated in layers 7–8. Layers 1–6 show near-zero gap — same-class and diff-class pairs are nearly indistinguishable in early activations. This directly supersedes the "flat patching influence" finding from Part 2, which was an artifact of the cascade confound averaging over all downstream layers.

Note: this is also consistent with the Stage 1 baseline per-layer silhouette table (Part 1), which showed layers 1–4 with negative silhouette and the jump to positive values only at layers 7–8.

**Scalar trajectory similarity (mean across layers):**

| Model | Same-class mean | Diff-class mean | Separation |
|-------|----------------|----------------|-----------|
| Option A | 0.848 | 0.713 | 0.135 |
| Option B | 0.841 | 0.709 | 0.132 |

Unlike the binary patching CircuitSim, the distributions are continuous and overlapping — the expected shape for a meaningful continuous ground truth.

---

### Step 3 (revised) — z-Space Proxy Validation

#### Primary correlation

| Model | Spearman ρ | p-value | 95% bootstrap CI |
|-------|-----------|---------|-----------------|
| Option A (weighted_sum) | **0.797** | 6.58e-221 | [0.775, 0.816] |
| Option B (transformer_cls) | **0.781** | 6.29e-206 | [0.756, 0.801] |

Both exceed the validation threshold of ρ > 0.5 by a large margin. The bootstrap CIs exclude zero by more than 50 standard deviations — the correlation is not a sampling artifact. These values supersede the original ρ = 0.717/0.743 against the binary patching ground truth.

#### Per-layer ρ: which layers does z reflect?

For each layer l, Spearman ρ between z-space cosine similarity and sim_l:

| Layer | Option A ρ | Option B ρ |
|-------|-----------|-----------|
| 1 | 0.173 | 0.195 |
| 2 | 0.221 | 0.259 |
| 3 | 0.283 | 0.276 |
| 4 | 0.313 | 0.314 |
| 5 | 0.428 | 0.411 |
| 6 | 0.625 | 0.611 |
| 7 | 0.857 | 0.862 |
| 8 | **0.913** | **0.912** |

z-space similarity correlates with layer 8 at ρ = 0.91 — higher than the overall ρ of 0.797 against mean trajectory similarity. This is only possible if z is dominated by layers 7–8, with earlier layers contributing marginal signal. The meta-encoder has learned to be approximately a function of the last two blocks.

#### Baseline comparison: does z add over single-layer proxies?

Spearman ρ vs. pair label (1 = same-class, 0 = diff-class):

| Signal | Option A | Option B |
|--------|----------|----------|
| z-space cosine sim | 0.837 | 0.845 |
| h₈ cosine sim (last layer) | **0.846** | **0.846** |
| h₁ cosine sim (first layer) | 0.137 | 0.154 |

**h₈ alone matches or marginally exceeds z for class discrimination.** The full trajectory encoder — all projectors, meta-encoder weights, InfoNCE training — adds no measurable benefit over raw last-layer cosine similarity on this task. The multi-layer machinery has converged to a solution that is effectively a learned transform of h₈.

---

### Augmentation Invariance

K=5 augmented views (random crop, horizontal flip, color jitter) of N=200 validation images. Within-image z-similarity = mean pairwise cosine similarity across the K views.

| Condition | Option A | Option B |
|-----------|----------|----------|
| Within-image (augmented views) | 0.930 | 0.914 |
| Same-class pairs (Section 3) | 0.937 | 0.928 |
| Diff-class pairs (Section 3) | 0.289 | 0.173 |

Within-image ≈ same-class. The expected ordering (within-image > same-class > diff-class) is not satisfied — augmented views of one image are no more similar to each other than entirely different same-class images. This means z cannot distinguish instance identity within a class; it encodes class membership only. The InfoNCE objective with class-label positive pairs has caused class-level collapse: every image of a given class maps to approximately the same region of z-space, regardless of which specific dog or airplane it is.

---

### Architectural Finding: End-to-End Gradient Flow

Both backbone and meta-encoder are updated by a single AdamW optimizer over all parameters jointly. The trajectory tensors are not detached before being passed to the meta-encoder. The InfoNCE gradient flows: z → projectors → backbone activations → backbone weights.

Because the meta-encoder weights layers 7–8 most heavily (depth ramp or learned attention), the gradient reaching the backbone is also concentrated in layers 7–8. The backbone reorganizes those layers most aggressively, making them more class-discriminative. This is a self-reinforcing cycle: stronger layer-8 gradient → layer 8 becomes more discriminative → meta-encoder weights layer 8 more → stronger gradient.

This explains the h₈ ≈ z result: both the backbone and meta-encoder have jointly converged toward a solution dominated by the last two blocks. The end-to-end design is correct and working as intended — the question is whether the training signal (class-label positive pairs) is the right one to produce the intended circuit structure.

---

## Part 4 — Cross-Cutting Findings

### Option A vs. Option B

In the revised validation (Part 3), Option A achieves slightly higher ρ (0.797 vs 0.781), reversing the original ordering from Part 2 (0.717 vs 0.743). Both are well within bootstrap CI overlap, so neither is definitively superior. The revised ground truth (continuous per-layer similarity) is more informative than the original binary patching measure, so the Part 3 numbers are the more reliable comparison.

The per-layer ρ profiles are nearly identical for both models (layer 8: 0.913 vs 0.912). Option B's learned attention provides no measurable advantage in terms of which layers get weighted — both converge to the same late-layer-dominant solution. The tradeoff from the original analysis remains: Option A's z is directly auditable (fixed composition); Option B's varies per input. But the gap is not meaningful at the current evaluation scale.

### The Depth Ramp Finding: Revised

The original finding ("depth ramp is miscalibrated; empirical patching influence is flat") was an artifact of the cascade confound. The revised per-layer gap table (Part 3) shows the opposite pattern: class-discriminative structure increases sharply with depth, with gap ratios of 35:1 between layer 8 and layer 1.

**Revised interpretation:** The depth ramp direction is correct (weight later layers more), but it significantly undershoots the empirical structure. The linear ramp gives layer 8 a weight of ~8× layer 1 (0.22 vs 0.028). The empirical gap ratio is ~35× (0.525 vs 0.015). A steeper ramp — or learned weighting — would better match the actual per-layer discriminative structure of ResNet18 on CIFAR-10.

Whether a steeper ramp would improve ρ, or whether the gap would simply be absorbed into h₈ dominance regardless of weighting, is an open question.

### Class-Level Collapse

The most important new finding from Part 3 is class-level collapse: within-image z-similarity (0.930) equals same-class pair z-similarity (0.937). z cannot distinguish instance identity within a class. The InfoNCE objective with class-label positive pairs has driven the backbone and meta-encoder to a solution where every image of a given class maps to approximately the same point in z-space.

This is not a failure of the implementation — it is the rational response to the training signal. Class-label positive pairs define "same circuit" as "same class." The model optimized for that definition and achieved it. The intended goal — capturing finer-grained within-class circuit variation — requires a different positive pair definition that reflects actual circuit similarity rather than semantic category.

### z-Space Mean Similarity

z-space mean cosine similarities (Part 2 experiment): Option A = 0.558, Option B = 0.527.

Both are well above zero (collapsed embeddings approach 1.0; random unit vectors in 64 dimensions approach 0). The InfoNCE objective produces a well-spread embedding space without explicit regularization on individual per-layer projections.

---

## Part 5 — Complete Results Summary

| Metric | Original CTLS | Option A (unified) | Option B (unified) |
|--------|--------------|-------------------|-------------------|
| Circuit silhouette | 0.8097 | 0.808 | 0.823 |
| Output silhouette | 0.8124 | 0.813 | 0.827 |
| Intraclass ρ (mean) | 0.768 | 0.748 | 0.722 |
| Noise ratio (σ=0.3) | 0.784 | 0.667 | 0.783 |
| Proxy ρ vs. per-layer traj sim | — | **0.797** | **0.781** |
| Proxy ρ 95% CI | — | [0.775, 0.816] | [0.756, 0.801] |
| Proxy ρ vs. layer 8 only | — | 0.913 | 0.912 |
| z-sim vs. h₈-sim (disc. power) | — | 0.837 vs 0.846 | 0.845 vs 0.846 |
| Within-image aug invariance | — | 0.930 | 0.914 |
| Same-class z-sim mean | — | 0.937 | 0.928 |
| Diff-class z-sim mean | — | 0.289 | 0.173 |
| (deprecated) Proxy ρ vs. patching CircuitSim | — | 0.717 | 0.743 |

**Key findings summary:**

| Finding | Evidence |
|---------|---------|
| Unified objective replicates original metrics | Circuit silhouette 0.82 vs 0.81 original |
| Proxy validated (revised, continuous GT) | ρ = 0.797/0.781 vs per-layer trajectory similarity |
| Bootstrap CIs exclude zero | 95% CI [0.775, 0.816] / [0.756, 0.801] |
| Class-discriminative structure is late-layer dominated | Per-layer gap: 0.015 at layer 1, 0.525 at layer 8 |
| z dominated by layers 7–8 | Per-layer ρ: 0.913 at layer 8, 0.173 at layer 1 |
| z ≈ h₈ in discriminative power | ρ vs labels: z = 0.837, h₈ = 0.846 (Option A) |
| Class-level collapse from class-label pairs | Within-image sim (0.930) ≈ same-class sim (0.937) |
| Backbone trained end-to-end; gradient concentrated in layers 7–8 | No detach() on trajectory; joint optimizer |
| Depth ramp direction correct, magnitude understated | Gap ratio 35:1 empirically vs 8:1 in ramp |
| CTLS accuracy improves over baseline | 94.21% vs 93.53% |
| Output-space structure preserved | Output silhouette: +0.003 over baseline |
| CTLS uses structured superposition | Δmono = −0.08, Δsilhouette = +0.66 |

---

## Open Questions (Steps 4–6)

- **Step 4:** Does a different activation extraction strategy (pre-nonlinearity, spatially-resolved, gradient-weighted) produce higher proxy ρ against per-layer trajectory similarity ground truth?
- **Step 5 (now most urgent):** Does defining positive pairs by actual trajectory similarity (rather than class label) break class-level collapse, force the backbone to reorganize earlier layers, and produce a z that adds meaningfully over h₈? The h₈ ≈ z finding makes this the critical next experiment.
- **Step 6:** Does CTLS-SSL outperform DINO/SimCLR on sample efficiency for semantically related new categories?
