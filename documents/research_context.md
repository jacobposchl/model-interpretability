# CTLS Research Context
### Circuit Trajectory Latent Space — Project Overview and Scientific Grounding

---

## 1. The Core Problem: Interpretability as an Afterthought

Most neural network interpretability research asks: *how do we understand what a trained model learned?* Sparse autoencoders are fit to activations. Probing classifiers are layered on top of frozen representations. Attention maps are visualized. These are all post-hoc methods — they treat a trained model as an archaeological artifact to be reverse-engineered after the fact.

This creates three fundamental problems:

**Faithfulness gap.** Post-hoc explanations may not accurately reflect the model's actual computational process. An SAE or probing classifier explains what a fixed-point analysis can see, not necessarily what the model is doing during inference.

**Spurious circuits.** Because circuits are discovered rather than enforced, the same model may use entirely different internal pathways for semantically identical inputs. Explanations are unstable: run the same post-hoc tool on two images of the same class and you may recover completely different "circuits."

**No feedback loop.** Post-hoc analysis cannot change how the model computes. It can only describe it. There is no mechanism for using interpretability insights to make the model more interpretable going forward.

**CTLS treats interpretability as a training-time design constraint rather than a post-hoc analysis tool.** The model is not just trained to produce correct outputs — it is trained to produce correct outputs *via consistent, structured internal pathways*.

---

## 2. Biological Motivation: Population Dynamics, Not Single Neurons

The conceptual foundation of CTLS comes from systems neuroscience. The brain does not encode information in single neurons. Individual neurons are noisy, unreliable, and often respond to multiple unrelated stimuli. Information is encoded in **population dynamics** — the specific pattern of co-activation across many neurons simultaneously.

This is why neuroscientists use dimensionality reduction techniques like PCA, UMAP, and trajectory analysis on population recordings rather than analyzing individual neurons in isolation. A concept like "the animal is running" is not stored at one address — it is a trajectory through neural population space.

The key quantitative insight: with N neurons, studying neurons in isolation gives you N dimensions. But the space of possible population activation patterns has approximately 2^N configurations — exponentially larger. This vast representational capacity means a network has no need to "cheat" by overlapping unrelated concepts onto the same neurons (superposition) when it has population space to work with.

CTLS borrows this directly. By organizing the model's population-level activation trajectories rather than individual neuron activations, the method targets the right unit of analysis. The experimental results confirm this reasoning: CTLS achieves high circuit-level organization through structured population patterns, and this comes with *lower* individual-neuron monosemanticity — because the model is using population space rather than packing separate concepts into single neurons.

---

## 3. What Is a "Circuit" in This Project?

The strict mechanistic interpretability definition of a circuit is the causal subgraph of the weight matrix responsible for a specific computation — identified via activation patching and causal interventions. This is the ground truth, but it is expensive to compute and non-differentiable; it cannot serve as a training signal.

**CTLS uses the activation trajectory as a differentiable proxy for circuit identity.**

For an input `x` passed through a model with `L` layers, the activation trajectory is:

```
T(x) = (h₁(x), h₂(x), ..., h_L(x))
```

where `h_l(x)` is the (globally average-pooled) activation vector at layer `l`. This is cheap, differentiable, and computable during training.

**The central bet:** if two inputs share a causal circuit — if they route through the same computational subgraph — then their activation trajectories should be geometrically similar across all layers. Trajectory similarity is a valid proxy for circuit similarity.

This bet has been empirically validated. The validation experiments (Steps 1–3) show Spearman ρ = 0.71–0.74 between z-space cosine similarity and activation-patching ground-truth circuit similarity. See `documents/results.md` for the full validation analysis.

---

## 4. The Unified Architecture

### 4.1 Why a Unified Objective?

An earlier formulation of CTLS used two separate loss signals operating on two different objects — a depth-weighted cosine loss on raw activations, and a supervised contrastive loss on meta-encoder outputs. These were architecturally disconnected: depth weighting never touched the meta-encoder, and the meta-encoder had no structural incentive to care about depth. The current unified objective corrects this by making depth weighting a geometric property of how the circuit embedding `z` is *constructed*, not an external loss coefficient bolted onto something else.

### 4.2 Step 1 — Per-Layer Projection to a Common Dimension

The backbone (ResNet18) has layers of varying widths (64 → 64 → 128 → 128 → 256 → 256 → 512 → 512 for the 8 ResNet blocks). Before any combination can happen, each layer's activation vector must be projected to a common dimension `d`:

```
p_l = LayerNorm(GELU(Linear_l(h_l)))    where p_l ∈ R^d for all l
```

Each `Linear_l` is a separate learned projector for layer `l`. This makes the meta-encoder architecture-agnostic — the same design works for any backbone because width differences are absorbed by the per-layer projectors before the combination step.

### 4.3 Step 2 — Depth-Aware Combination into z

The projected sequence `[p₁, ..., p_L]` is combined into a single circuit embedding `z`. Two variants are implemented:

**Option A — `weighted_sum` (fixed linear depth ramp):**

```
z = Σ_l w_l · p_l    where w_l = l / Σ(1..L)
```

followed by a final linear layer and L2 normalization. The depth weights are fixed and not learned — layer 1 gets weight 1/36, layer 8 gets weight 8/36 (for L=8). Every `z` vector has the same composition structure regardless of input: two inputs' z-similarity is a fixed linear combination of their per-layer representational similarities, with deeper layers contributing more. This makes distances in z-space directly interpretable.

**Option B — `transformer_cls` (learned depth attention):**

The projected sequence `[CLS, p₁, ..., p_L]` has sinusoidal depth-encoding positional embeddings added to each `p_l`. A 2-layer transformer encoder processes the sequence. The CLS token output is projected and L2-normalized to produce `z`. This allows the model to learn *which layers matter per input* rather than applying a fixed ramp to everyone. For an image pair where class identity becomes clear at layer 5, attention can weight layer 5 more; for a harder pair needing layer 7, it weights layer 7 more.

The tradeoff: Option A's `z` has fixed, interpretable composition. Option B's composition varies per input, making cross-input distance comparisons harder to interpret uniformly.

**Validation finding:** Option B slightly outperforms Option A on proxy validation (ρ = 0.743 vs 0.717), likely because the dominant patching layer varies per pair with no consistent depth bias — Option B's per-input attention weights can adapt to this, a fixed ramp cannot.

### 4.4 Step 3 — Single InfoNCE Loss on z

```
L_cons = −log( exp(cos(z_i, z_j) / τ) / Σ_k exp(cos(z_i, z_k) / τ) )
```

where `z_i` is the anchor circuit embedding, `z_j` is the positive (same class, different image), and the denominator sums over all other images in the batch as negatives. `τ = 0.07` (InfoNCE temperature).

**Why InfoNCE over MSE:** The loss uses cosine similarity rather than MSE. MSE between normalized vectors is divided by dimensionality D, meaning 512-dim later layers would contribute ~8× less signal than 64-dim early layers — the opposite of what depth-weighting intends. Cosine distance is dimension-independent, always returning a value in [0, 1] regardless of layer size.

**Why different same-class images, not augmentations:** Same-image augmentations produce trajectories that are already similar before any training — the loss would be trivially satisfied from initialization. Different same-class images require genuine semantic alignment across the trajectory: the model must learn to route a close-up photo of a cat and a distant photo of a cat through similar internal pathways, which is the actual training signal.

**Total training objective:**

```
L_total = L_task + λ · L_cons
```

where `L_task` is standard cross-entropy and `λ` is warmed up linearly from 0 to 1 over the first 10 epochs to avoid disrupting backbone representations before they stabilize.

### 4.5 Soft Masking

Activations use magnitude-weighted soft gates rather than binary masks:

```
S_i = σ(a_i / τ_mask) · a_i
```

Temperature `τ_mask` is cosine-annealed from 1.0 to 0.1 over training — starting fluid (near-linear gating), hardening toward near-binary, interpretable circuits by the end of training. This is separate from the InfoNCE temperature `τ = 0.07`.

---

## 5. Theoretical Grounding and Intuition

### 5.1 Why Depth-Weighting?

The feature hierarchy in deep networks is not uniform. Early layers encode surface features — edges, textures, local shapes — that legitimately vary across instances of the same semantic category. A close-up photo of a cat and a distant one look very different at the pixel level, and layers 1–3 of a ResNet encode this low-level variation. It would be counterproductive to force these early representations to be similar.

Later layers encode abstract semantic content. By layer 7–8 of ResNet18, the representation captures "cat" rather than "this particular arrangement of pixels." These are the layers that should be consistent for same-class inputs.

The depth ramp implements this prior structurally: consistency pressure increases with layer depth, letting early layers develop naturally while organizing the semantically loaded later representations.

**Empirical caveat:** The activation patching validation found that causal influence is approximately flat across all 8 layers on average — no layer dominates causally. The depth ramp is therefore better understood as a useful training prior rather than a description of where computation actually happens. Whether a uniform or learned weighting would produce higher ρ against patching ground truth is an open empirical question (Step 4 of the roadmap).

### 5.2 Why Circuit Space Differs from Output Space

The output embedding is optimized to be maximally useful for the task. It discards everything that does not contribute to predicting the correct label — it is the endpoint of computation.

The circuit latent space encodes the trajectory: every intermediate step that led to that endpoint. Two inputs can produce identical output embeddings (both correctly classified as "cat") while having taken completely different internal routes — different early-layer feature processing, different attentional routing, different intermediate representations. The circuit latent space captures this difference; the output space does not.

This is not just theoretical. The Stage 2 results show it directly: output silhouette remains essentially unchanged at 0.81 while circuit silhouette jumps from 0.15 to 0.81. CTLS did not achieve circuit organization by collapsing the backbone into encoding final-layer-like features at every layer. The model learned a different way to represent class identity in its trajectory without disrupting classification.

### 5.3 Connection to Grokking and MoE Routing

Research on grokking in Mixture-of-Experts models shows that routing pathways spontaneously become more consistent across same-category inputs at the generalization transition — before grokking, routing is noisy; after grokking, the same inputs consistently use the same experts. The CTLS consistency loss may be accelerating this same transition by providing it as a direct training signal rather than waiting for it to emerge spontaneously.

---

## 6. What Has Been Established Experimentally

The experimental work falls into two phases:

**Phase 1 — Original Stage Experiments (Stages 1–5):**
Five experiments on CIFAR-10 / ResNet18 characterized the original CTLS objective and established foundational metrics. Key findings: circuit silhouette 0.15 → 0.81 (+0.66), accuracy 93.53% → 94.21% (+0.68%), output silhouette unchanged (+0.003), depth-weighting outperforms uniform on all metrics, and the monosemanticity paradox (higher circuit organization but lower individual neuron monosemanticity). See `documents/results.md` for the full breakdown.

**Phase 2 — Validation Experiment (Steps 1–3):**
The unified architecture was implemented and validated against an independent causal ground truth. Three things were established:

1. **Unified objective replicates and slightly exceeds original CTLS metrics.** Circuit silhouette 0.819, output silhouette 0.824, essentially identical to the original two-signal system but with a cleaner, internally consistent architecture.

2. **Proxy validated at ρ = 0.71–0.74.** Trajectory cosine similarity tracks activation-patching circuit similarity. Crucially, the within-same-class ρ (0.709) is essentially identical to the overall ρ — the correlation is not inflated by trivial between-class separation. Among images from the same class, `z`-space correctly identifies which pairs share their circuit routing versus which take different internal paths to the same prediction.

3. **~50% of same-class image pairs do not share circuits.** Class label and circuit identity are not the same thing. Two correctly-classified "cat" images can use completely different internal pathways. This finding directly motivates using patching-derived positive pairs rather than class-label positive pairs as the training signal (Step 5).

---

## 7. The Monosemanticity Paradox

The most theoretically significant finding: CTLS achieves dramatically higher circuit silhouette (0.81 vs 0.15) but *lower* monosemanticity across all layers (Δmono = −0.080 average). This apparent contradiction is worth understanding.

The standard monosemanticity metric counts what fraction of SAE dictionary features activate selectively for a single class. The baseline has higher monosemanticity — yet its circuit silhouette is 0.15, meaning its clusters are nearly unstructured. How can a model with more monosemantic features be less class-organized in embedding space?

The resolution: baseline "monosemantic" features are monosemantic *by chance on a noisy background*. Because the baseline's activations have no class-structure at the trajectory level, the SAE is fitting essentially random variance. Some features in that noise happen to correlate with one class — these count as monosemantic. But they are sparse islands of class signal in a sea of noise.

CTLS encodes class identity through **structured superposition** — features that activate for multiple related classes simultaneously (a "fur texture" feature active for both cat and dog; a "wings" feature active for bird and airplane). No individual feature is exclusively class-specific, so monosemanticity is low. But the *combination* of features encodes class identity with high fidelity — which is why circuit silhouette is high.

This is the population dynamics insight made concrete: the model uses the exponentially large population space rather than forcing monosemantic single-neuron encoding. CTLS does not eliminate superposition — it *structures* it so that the superposition patterns are consistent within a class.

**Implication for evaluation methodology:** SAEs are a neuron-level analysis tool. CTLS never optimized for neuron-level monosemanticity — it optimized for population-level trajectory consistency. Using SAEs as the primary evaluation of CTLS is analogous to evaluating a symphony by checking whether each instrument plays only one note. The appropriate evaluation tools are trajectory-level: circuit silhouette, intraclass Spearman ρ, noise robustness ratio, and — most importantly — Spearman correlation against activation-patching ground truth.

---

## 8. Novelty and Relationship to Prior Work

| Method | Mechanism | What It Misses |
|--------|-----------|----------------|
| Sparse Autoencoders (SAEs) | Post-hoc single-layer decomposition | Training-time objective; multi-layer trajectories; cross-input consistency |
| Monosemantic Feature Neurons (MFNs) | Stability loss on bottleneck under noise | Cross-input semantic consistency; full trajectory; depth-weighting |
| MonoLoss | Differentiable monosemanticity score per neuron | Circuit routing; population dynamics; trajectory embedding |
| Activation Consistency Training (ACT) | Residual stream consistency under prompt perturbation | Interpretability goal; semantic grouping; circuit extraction |
| Brain-Inspired Modular Training (BIMT) | Weight-level penalty on connection length | Activation trajectories; training-time consistency loss |
| CKA / Probing Classifiers | Post-hoc representation similarity analysis | Training-time objective; any feedback to model |
| **CTLS (this project)** | Joint latent space over full multi-layer activation trajectories with depth-aware InfoNCE semantic consistency | — |

The specific combination that does not appear in existing literature: a training-time objective that (1) treats the full multi-layer activation trajectory as the unit of analysis, (2) embeds that trajectory into a depth-aware joint latent space, and (3) uses contrastive semantic consistency between genuinely distinct same-category inputs as the loss signal — as opposed to perturbation-based stability of the same input under noise or augmentation.

---

## 9. Scientific Applications

**Medical imaging and clinical AI.** CTLS solves a specific problem: *reasoning consistency*. A diagnostic model might flag two similar chest X-rays for the same risk level for entirely different internal reasons — one genuinely pathological, the other triggered by an imaging artifact. Post-hoc methods cannot reliably distinguish these cases. A CTLS-trained model provides a structural guarantee that similar pathologies activate similar circuits. When a new scan lands far from the expected circuit cluster for its predicted class, that deviation is automatically flagged — uncertainty quantification at the circuit level, not just the output level.

**AI safety and alignment.** One core problem in safety is detecting when a model reasons differently than it appears to — using an internal pathway that bypasses circuits researchers deemed safe. CTLS makes this detectable by construction. Deviation from expected circuit clusters is measurable in z-space. The 0.784 vs 0.273 noise robustness ratio at high noise suggests this signal holds even under significant distribution shift.

**Continual learning and domain adaptation.** Catastrophic forgetting happens partly because new training restructures circuits that were working for old data. CTLS provides a concrete mechanism for monitoring and preventing this — tracking whether new training disrupts the circuit structure of previously-learned categories.

---
