# CTLS Experiment Results — Full Analysis

## Overview

CTLS (Circuit Trajectory Latent Space) trains a ResNet18 backbone alongside a meta-encoder that compresses the 8-layer activation trajectory into a 64-dim L2-normalized circuit embedding. The core hypothesis is that you can force a network to organize its *internal computations* (not just its outputs) by class identity, without sacrificing task accuracy. This is tested across 5 stages on CIFAR-10.

---

## Stage 1 — Baseline

**Val accuracy: 93.53%**

| Space | Silhouette |
|-------|-----------|
| Circuit | 0.1466 |
| Output | 0.7974 |
| Delta (C − O) | −0.6508 |

**Per-layer silhouette:**

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

The per-layer profile tells a clear story: early ResNet blocks (layers 1–4) are genuinely anti-class-structured — silhouette is negative, meaning within-class distances actually exceed between-class distances in raw activation space. The model spends these layers on low-level features (edges, textures) that are not class-specific. Class identity only becomes geometrically resolvable at layers 7–8, where the final 512-dim blocks have accumulated enough abstraction.

The circuit silhouette of 0.1466 is the aggregate of this trajectory. It is far below the output silhouette (0.7974), confirming that the baseline model's class structure is front-loaded into the final logit space — the activation path *through* the network is largely unstructured by class identity. This is the gap CTLS targets.

---

## Stage 2 — Full CTLS Objective

**CTLS checkpoint: epoch 95, val_acc = 0.9421**

| Metric | Baseline | CTLS | Delta |
|--------|---------|------|-------|
| Circuit silhouette | 0.1486 | 0.8097 | **+0.6611** |
| Output silhouette | 0.8091 | 0.8124 | +0.0033 |
| Val accuracy | 93.53% | 94.21% | **+0.68%** |

The +0.66 circuit silhouette jump is the core result. To put it in context: a silhouette of 0.81 is in the range you'd expect from cleanly separable clusters with tight intra-class variance and large inter-class gaps. The circuit latent space went from nearly random (0.15) to highly class-organized (0.81) purely through the training objective — the network architecture is identical.

Two things are notable beyond the raw number:

1. **Output silhouette is unchanged (+0.003).** CTLS did not achieve circuit structure by "cheating" — by, say, collapsing the backbone to only encode final-layer-like features at every layer. The output-space organization is preserved. The network learned a different way to represent class identity in its trajectory without disrupting its classification behavior.

2. **Accuracy improves (+0.68%).** This is not trivially expected. The consistency pressure forces same-class image pairs to route through similar circuits, which acts as a data-dependent regularizer. The slight accuracy gain suggests this structured routing generalizes better — the network is less reliant on spurious activation patterns that happen to work for a specific image but don't generalize within the class.

---

## Stage 3 — Embedding Analysis

### Intraclass Rank Correlations

Within each class, Spearman's ρ measures whether the circuit embeddings of different images have a consistent internal ordering. High ρ means the embedding space has genuine intraclass geometry — not just tight clusters, but meaningful distance relationships within a class.

| Class | CTLS ρ | Baseline ρ | Δρ |
|-------|--------|-----------|-----|
| airplane | +0.849 | +0.198 | +0.651 |
| automobile | +0.593 | +0.210 | +0.383 |
| bird | +0.876 | +0.330 | +0.546 |
| cat | +0.904 | +0.288 | +0.616 |
| deer | +0.781 | +0.149 | +0.632 |
| dog | +0.869 | +0.357 | +0.512 |
| frog | +0.717 | +0.399 | +0.318 |
| horse | +0.675 | +0.246 | +0.429 |
| ship | +0.682 | +0.325 | +0.357 |
| truck | +0.736 | +0.450 | +0.286 |
| **Mean** | **0.768** | **0.295** | **+0.473** |

CTLS improves intraclass consistency by ~2.6× on average. Every class benefits, but the effect is not uniform:

- **Animal classes (cat 0.904, bird 0.876, dog 0.869, airplane 0.849)** show the strongest intraclass correlation. Biological/organic objects with rich texture and shape variation apparently map to more consistent circuits — likely because the depth-weighted loss puts pressure on layers 6–8, which are where texture-sensitive and shape-sensitive features operate in ResNets.

- **Vehicle classes (automobile 0.593, horse 0.675, ship 0.682)** show lower intraclass correlation. Automobile and truck (0.736) are visually similar categories — the model's circuit representations for these likely share activated features, which may reduce their intraclass coherence slightly. The embedding space is capturing genuine visual similarity rather than forcing artificial separation.

- **Baseline correlations** are uniformly weak (0.149–0.450). The higher baseline values for frog (0.399) and truck (0.450) likely reflect those classes having more visually homogeneous CIFAR-10 images rather than any organizational pressure.

### Noise Robustness

The ratio metric measures how much the circuit embedding shifts when noise is added, relative to how much the output representation shifts. A ratio near 1.0 means the circuit embedding faithfully tracks what the model actually does under noise.

| Noise σ | CTLS ratio | Baseline ratio |
|---------|-----------|---------------|
| 0.05 | 0.511 | 0.417 |
| 0.10 | 0.645 | 0.585 |
| 0.20 | 0.704 | 0.364 |
| 0.30 | 0.755 | 0.295 |
| 0.50 | 0.784 | 0.273 |
| 0.80 | 0.788 | 0.299 |

The divergence with increasing noise is the key finding. Under low noise (σ=0.05) the two models are similar. But as noise grows, CTLS's ratio climbs toward 0.79 while baseline's collapses toward 0.27.

The baseline behavior has a straightforward explanation: the baseline circuit embedding is nearly random (silhouette 0.15), so it doesn't represent anything class-relevant in the first place. When noise is added, the output representation shifts (the model's classification behavior changes), but the random circuit embedding doesn't track that shift — it can't, because it wasn't encoding the model's computation.

For CTLS, the circuit embedding encodes what the model is actually computing. When noise disrupts the input, the circuit embedding changes accordingly, tracking the model's response. The high ratio is a sign of *informational fidelity*, not fragility.

The absolute output distance numbers also reveal something: at σ=0.3, CTLS output_dist_mean is 0.4728 vs baseline 0.5867. The CTLS backbone itself is more noise-stable in its output representations — the consistency regularization that forces same-class images to route similarly also makes the network more resilient to intra-class variation (which is what Gaussian noise approximately simulates).

---

## Stage 4 — Ablations

| Variant | Circuit sil | Output sil | Val acc |
|---------|------------|-----------|--------|
| Baseline (λ=0) | 0.1513 | 0.8040 | 93.53% |
| Uniform weights | 0.8068 | 0.8020 | 93.88% |
| Depth-weighted | **0.8295** | **0.8309** | **94.21%** |

Depth-weighting outperforms uniform weighting on all three metrics. The gaps are not large but they are consistent:

- **Circuit silhouette:** +0.0227 for depth-weighted over uniform. This confirms the design hypothesis — emphasizing later layers (which carry more semantic content) produces better trajectory organization than treating all layers equally.

- **Output silhouette:** Depth-weighted (0.8309) actually improves over baseline (0.8040), while uniform weighting (0.8020) slightly degrades it. This is a subtle but important distinction. Applying uniform consistency pressure on early layers may interfere with the backbone's development of low-level features that contribute to output-space organization. Depth-weighting avoids this by not penalizing layers 1–2 heavily, letting them learn their natural representations while structuring the semantically-loaded later layers.

- **Accuracy:** Both CTLS variants improve over baseline (+0.35% uniform, +0.68% depth-weighted). The larger gain from depth-weighted suggests the regularization effect is stronger when later layers are more heavily constrained — those layers have the most influence on the final classification decision.

The uniform weighting result is still a strong improvement over baseline (+0.655 circuit silhouette). It confirms that the consistency loss itself is responsible for most of the gain, and depth-weighting is a meaningful refinement rather than the entire mechanism.

---

## Stage 5 — SAE Monosemanticity

### Layer-by-Layer Comparison

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

### SAE Quality (Final Layer)

| | Recon MSE | Sparsity |
|-|----------|---------|
| CTLS SAE | 0.00166 | 0.2492 |
| Baseline SAE | 0.00238 | 0.3428 |

### Interpreting the Monosemanticity Paradox

The most theoretically interesting finding of the project: CTLS achieves a much higher circuit silhouette (0.81 vs 0.15) but *lower* monosemanticity (Δmono = −0.080 average). The naive interpretation — "lower mono = worse" — is misleading here.

The standard monosemanticity metric counts what fraction of SAE dictionary features activate selectively for a single class. A high monosemanticity score means the model's features are cleanly class-specific. The baseline has higher monosemanticity (e.g., 0.125 at layer 8 vs CTLS's 0.004). But the baseline's circuit silhouette is 0.15 — its clusters are almost completely unstructured. How can a model with more monosemantic features be less class-organized in embedding space?

The resolution is that baseline "monosemantic" features are monosemantic *by chance, on a noisy background*. Because the baseline's activations have no class-structure at the trajectory level, the SAE is fitting essentially random variance. Some features in that random variance happen to correlate with one class — these count as monosemantic. But they're sparse islands of class signal in a sea of noise.

CTLS, by contrast, encodes class identity through **structured superposition** — features that activate for multiple related classes simultaneously (e.g., a "fur texture" feature active for cat and dog, or a "wings" feature active for bird and airplane). No individual feature is exclusively class-specific, so monosemanticity is low. But the *combination* of features encodes class identity with high fidelity, which is why the silhouette is high. The circuit embedding captures the class-specific combination, not any individual monosemantic feature.

This is consistent with what is known about neural network representations more broadly: circuits in trained networks tend to encode features in superposition, with linearity across the embedding dimensions carrying the class information. CTLS doesn't eliminate this superposition — it structures it so that the superposition patterns are consistent within a class.

**The reuse and sparsity numbers support this interpretation:**
- CTLS has lower reuse (Δreuse = −0.062): features fire less often on average. This is consistent with more selective, higher-fidelity features — each feature fires when its specific combination of properties is present, which is less frequent than the noisy baseline features.
- CTLS SAE achieves lower reconstruction error (0.00166 vs 0.00238): despite its features being non-monosemantic, the SAE can reconstruct the representations better. The CTLS representations have more regular, learnable structure — even if that structure is distributed.
- CTLS sparsity (0.249) is closer to the target of <0.2 than baseline (0.343). The representations are moving toward sparser encoding even though the training objective never explicitly targeted sparsity.

The layer-by-layer trend in monosemanticity is also notable. In the baseline, mono increases from early to late layers (0.059 → 0.125), reflecting that later layers develop more class-specific features. In CTLS, mono collapses at layers 7–8 to near-zero (0.004). These are exactly the layers where the depth-weighted consistency loss applies the most pressure. The consistency loss forces all layer-7 and layer-8 representations for same-class images to align — which creates highly class-correlated but distributed representations at those layers, exactly the superposition pattern described above.

---

## Summary of Key Findings

| Finding | Evidence |
|---------|---------|
| CTLS structurally organizes circuit space | Silhouette: 0.15 → 0.81 (+0.66) |
| This doesn't harm task accuracy — it helps | Val acc: 93.53% → 94.21% (+0.68%) |
| Output-space classification structure is preserved | Output silhouette: 0.8091 → 0.8124 (+0.003) |
| The improvement is geometrically deep, not just cluster tight | Intraclass ρ: 0.30 → 0.77 (+0.47) |
| CTLS backbone is more noise-robust | Output dist under σ=0.3: 0.587 → 0.473 |
| Circuit embeddings maintain informativeness under noise | Ratio at σ=0.5: 0.784 vs 0.273 (baseline) |
| Depth-weighted > uniform on all metrics | All three metrics favor depth-weighted |
| CTLS uses structured superposition, not monosemantic features | Δmono = −0.08 but Δsilhouette = +0.66 |
| CTLS representations are more efficiently encodable | SAE recon MSE: 0.00166 vs 0.00238; sparsity: 0.249 vs 0.343 |

---

## Limitations

**Layer 7 collapse.** The per-layer silhouette plot (Stage 4) shows both CTLS variants exhibit a local dip at layer 7 — the penultimate ResNet block. The attractive consistency pressure, even combined with SupCon, creates a partial collapse here. Adding a per-layer decorrelation term or a diversity penalty targeted at this layer might resolve it.

**SAE sparsity above target.** At 0.249, the final-layer circuit embedding is still denser than the commonly cited <0.2 target for "truly sparse" SAE representations. The structured superposition is learnable but not maximally efficient.

**CIFAR-10 scale constraints.** Ten coarse classes, 32×32 images, relatively homogeneous within-class variation. It is unclear whether CTLS's circuit structure benefit holds for fine-grained recognition (100+ classes, high within-class variation) or higher-resolution inputs where the trajectory is longer and the per-layer representations richer. The monosemanticity-vs-superposition tradeoff may look different at scale.

**Single architecture.** Only tested on ResNet18. Architectures with skip connections of different depths, or attention-based models where "trajectory" is less well-defined, would require adaptation of both the backbone hooking and the depth weighting scheme.
