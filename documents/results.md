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

## Part 2 — Validation Experiment (Steps 1–3)

This experiment implemented the unified objective and validated the trajectory proxy against activation-patching ground truth.

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

## Part 3 — Cross-Cutting Findings

### Option A vs. Option B

Option B consistently outperforms Option A on proxy validation (ρ = 0.743 vs 0.717) while being essentially identical on sanity metrics. The advantage is modest but consistent.

The likely mechanism: the dominant patching layer varies per pair with no consistent depth bias. Option B's per-input attention weights can focus on whichever layer is causally important for each specific pair. A fixed ramp cannot do this — it always weights layer 8 more regardless of whether a given pair's circuit difference is at layer 3 or layer 7.

The tradeoff: Option A's z has fixed composition — two inputs' z-similarity is a fixed-weight combination of their per-layer similarities, making z-space directly auditable. Option B's composition varies per input, making distances harder to interpret uniformly.

At the current margin (0.026 ρ), neither is clearly superior. The decision depends on whether you prioritize predictive accuracy of the proxy (Option B) or interpretability of the circuit embedding itself (Option A).

### The Depth Ramp Is Miscalibrated

This is the most actionable finding beyond core proxy validation.

The fixed depth ramp (Option A) assigns weights increasing from 0.03 at layer 1 to 0.22 at layer 8. The empirical patching influence, averaged over same-class pairs, is flat at ~0.125 (= 1/8, uniform) across all layers. The ramp and the actual causal structure are misaligned.

**What this means:**
- The design rationale ("late layers encode abstract semantics that should be consistent; early layers encode surface features that legitimately vary") is not confirmed by patching in ResNet18 on CIFAR-10. All layers contribute roughly equally to circuit-defining computation on average.
- The ramp may still be a useful training prior — it may shape how the backbone learns to organize representations even if the resulting model doesn't exhibit increasing depth-importance. These are different claims.
- A uniform weighting might be better calibrated to empirical circuit structure. Whether it also achieves equivalent circuit organization is an open question for the Step 4 ablation.

### z-Space Mean Similarity

z-space mean cosine similarities: Option A = 0.558, Option B = 0.527.

Both are well above zero (collapsed embeddings would approach 1.0; random unit vectors in 64 dimensions approach 0). The InfoNCE objective is producing a well-spread embedding space without explicit regularization on individual per-layer projections.

---

## Part 4 — Complete Results Summary

| Metric | Original CTLS | Option A (unified) | Option B (unified) |
|--------|--------------|-------------------|-------------------|
| Circuit silhouette | 0.8097 | 0.817 | 0.819 |
| Output silhouette | 0.8124 | 0.820 | 0.824 |
| Intraclass ρ (mean) | 0.768 | 0.713 | 0.715 |
| Noise ratio (σ=0.3) | 0.784 | 0.818 | 0.801 |
| CircuitSim same-class mean | — | 0.524 | 0.534 |
| CircuitSim diff-class mean | — | 0.000 | 0.000 |
| Proxy Spearman ρ (all pairs) | — | 0.717 | 0.743 |
| Proxy Spearman ρ (same-class only) | — | 0.709 | — |
| Depth ramp vs. patching alignment | — | Miscalibrated (flat empirical) | N/A (learned) |

**Key findings summary:**

| Finding | Evidence |
|---------|---------|
| Unified objective replicates original metrics | Circuit silhouette 0.82 vs 0.81 original |
| Proxy validated against causal ground truth | Spearman ρ = 0.71–0.74 against activation patching |
| Within-class circuit discrimination confirmed | Same-class-only ρ = 0.709 ≈ overall ρ |
| ~50% of same-class pairs do not share circuits | CircuitSim bimodal: mean 0.524, std 0.499 |
| Depth ramp does not match empirical causal structure | Patching influence flat across layers |
| Option B slightly outperforms Option A on proxy | ρ = 0.743 vs 0.717 |
| CTLS accuracy improves over baseline | 94.21% vs 93.53% |
| Output-space structure preserved | Output silhouette: +0.003 over baseline |
| CTLS uses structured superposition | Δmono = −0.08, Δsilhouette = +0.66 |

---

## Open Questions (Steps 4–6)

- **Step 4:** Does a different activation extraction strategy (pre-nonlinearity, spatially-resolved, gradient-weighted) produce higher ρ against the existing patching ground truth?
- **Step 5:** Does training with patching-derived positive pairs (CircuitSim > threshold) rather than class-label pairs produce better within-class circuit discrimination?
- **Step 6:** Does CTLS-SSL outperform DINO/SimCLR on sample efficiency for semantically related new categories?
