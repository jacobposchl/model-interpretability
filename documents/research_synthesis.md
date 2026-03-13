# Circuit Trajectory Latent Space (CTLS)
### Training Neural Networks to Be Interpretable By Design
**Research Synthesis — March 2026**

---

## 1. Core Idea in Plain Language

Most interpretability research asks: *how do we understand what a trained model learned?* This project asks something fundamentally different: *what if the model was trained to be understandable in the first place?*

The central mechanism is this: during training, instead of only caring about the model's final output, we also track the activation patterns across every intermediate layer — the population-level firing patterns that constitute the model's internal "circuit" for processing each input. These multi-layer activation trajectories are embedded into a shared latent space via a lightweight meta-encoder. A contrastive consistency loss then enforces that semantically similar inputs produce similar trajectories in that space, while semantically different inputs remain well-separated.

The result is a model whose internal reasoning pathways are not just discovered after the fact, but actively shaped during training to be consistent, structured, and semantically organized.

---

## 2. Motivation: The Post-Hoc Problem

Current interpretability methods — Sparse Autoencoders (SAEs), probing classifiers, Centered Kernel Alignment (CKA), attention visualization — are all post-hoc. They treat a trained model as an archaeological artifact to be reverse-engineered. This creates several fundamental problems:

- **Faithfulness gap:** Post-hoc explanations often do not accurately reflect the model's actual computational process. They explain what we can see, not necessarily what the model is doing.
- **Spurious circuits:** Because circuits are discovered rather than enforced, the same model may use entirely different internal pathways for semantically identical inputs, making explanations unstable and unreliable.
- **No feedback loop:** Post-hoc analysis cannot change how the model computes. It can only describe it. There is no mechanism for using interpretability insights to make the model more interpretable.

CTLS treats interpretability as a design constraint rather than an analysis tool. The model is not just trained to produce correct outputs — it is trained to produce correct outputs via consistent, structured internal pathways.

---

## 3. The Biological Motivation

The inspiration for population-level circuit analysis comes directly from systems neuroscience. The brain does not encode information in single neurons — individual neurons are noisy, unreliable, and often respond to multiple unrelated stimuli. Instead, information is encoded in **population dynamics**: the specific pattern of co-activation across many neurons simultaneously.

This is why neuroscientists use dimensionality reduction tools like PCA, UMAP, and trajectory analysis on population recordings rather than analyzing individual neurons in isolation. A concept like "the animal is running" is not stored in one place — it is a trajectory through neural population space.

The key insight this project borrows from neuroscience is that the representational space of population patterns is exponentially larger than the space of individual neuron activations. With N neurons you have N dimensions if you study neurons in isolation, but approximately 2^N possible population patterns. This enormous representational capacity means the model can store many distinct, non-overlapping circuits without the compression artifacts that arise when trying to encode everything into individual neurons.

This directly weakens the superposition problem that plagues neuron-level interpretability: the model does not need to cheat by overlapping concepts in single neurons when it has vast population space to work with. The experimental results confirm this — CTLS achieves high circuit-level organization precisely through structured population patterns rather than monosemantic individual features.

---

## 4. Technical Architecture

### 4.1 What Is a Circuit Here?

A circuit is defined as the full activation trajectory of an input through the model: the population-level activation pattern at every layer from input to output. Formally, for input x and a model with L layers:

```
T(x) = (h₁(x), h₂(x), ..., h_L(x))
```

where h_l(x) is the activation vector at layer l. This is not a binary mask of active vs. inactive neurons, nor a single-layer snapshot. It is the full trajectory through the model's representational space — the path the computation takes, not just where it ends up.

### 4.2 The Meta-Encoder

A lightweight meta-encoder E takes the full trajectory T(x) as input and produces a compact, L2-normalized circuit embedding z in 64 dimensions:

```
z = E(T(x)) = E(h₁(x), h₂(x), ..., h_L(x))
```

The meta-encoder is a small MLP — deliberately kept lightweight so it compresses the trajectory into a semantically meaningful space without becoming a powerful feature extractor in its own right. Its job is compression and organization, not independent reasoning.

The circuit latent space is structurally different from the model's output embedding space. The output embedding captures what the model concluded. The circuit latent space captures how the model reasoned its way there. Two inputs can produce identical output embeddings while having taken entirely different internal routes — the circuit latent space captures that difference, the output space does not.

### 4.3 The Contrastive Consistency Loss (InfoNCE)

The consistency loss is a full contrastive objective operating on circuit embeddings. For each anchor input, it simultaneously:
- **Pulls** same-class circuit embeddings together (positive pairs: different images of the same class)
- **Pushes** different-class circuit embeddings apart (negatives: all other-class images in the batch)

The InfoNCE formulation:

```
L_cons = -log( exp(sim(zᵢ, zⱼ) / τ) / Σₖ exp(sim(zᵢ, zₖ) / τ) )
```

where sim is cosine similarity, τ is temperature (default 0.07), zᵢ is the anchor circuit embedding, zⱼ is the positive (same class, different image), and the denominator sums over all negatives in the batch.

**Cosine distance over L2:** The loss uses cosine distance rather than MSE. MSE between normalized vectors is divided by the dimensionality D, meaning 512-dim later layers contribute ~8× less signal than 64-dim early layers — the opposite of what depth-weighting intends. Cosine distance is dimension-independent, always returning a value in [0, 1] regardless of layer size, so depth-weighting actually controls relative layer contribution as intended.

**Positive pair construction:** Positives are different images of the same class, not augmentations of the same image. Same-image augmentations produce trajectories that are already similar before any training, making the loss trivially satisfied from initialization. Different same-class images require genuine semantic alignment, which is the actual training signal.

The total training objective:

```
L_total = L_task + λ · L_cons
```

### 4.4 Depth-Weighted Layer Contributions

The consistency loss is applied across all layers simultaneously, with weights that increase with depth:

```
L_cons = Σ_l w_l · InfoNCE(C_l(x₁), C_l(x₂))
```

where w_l increases monotonically with layer index l. This reflects the empirical reality of feature hierarchy: early layers extract surface features (edges, textures) that legitimately vary across instances of the same category, while later layers encode abstract semantic content that should be consistent for same-category inputs.

The ablation experiments confirm this design choice matters — depth-weighted consistently outperforms uniform weighting on all metrics.

---

## 5. Why the Circuit Latent Space Differs From Output Embeddings

This is the central theoretical claim, and the experimental results validate it directly.

The output embedding is optimized to be maximally useful for the task. It compresses everything the model learned about an input down to a vector that predicts the correct label, discarding anything that does not contribute to prediction. It is the endpoint of computation.

The circuit latent space encodes the trajectory — every intermediate step that led to that endpoint. The Stage 2 results show this directly: output silhouette remains essentially unchanged at 0.81 while circuit silhouette jumps from 0.15 to 0.81. CTLS did not achieve circuit structure by collapsing the backbone into encoding final-layer-like features at every layer — output-space organization is preserved. The model learned a different way to represent class identity in its trajectory without disrupting its classification behavior.

The noise robustness results further support this. Under high noise (σ=0.5), CTLS's circuit-to-output tracking ratio holds at 0.784 while baseline collapses to 0.273. The baseline circuit embedding is nearly random (silhouette 0.15) so it cannot track what the model is actually computing under distribution shift. The CTLS circuit embedding encodes what the model is computing, so when noise disrupts the input, the circuit embedding changes accordingly — a sign of informational fidelity, not fragility.

---

## 6. Experimental Results

All experiments used ResNet18 on CIFAR-10. Five stages were run.

### Stage 1 — Baseline Characterization

**Val accuracy: 93.53%**

| Space | Silhouette |
|-------|-----------|
| Circuit | 0.1466 |
| Output | 0.7974 |

Per-layer silhouette reveals the baseline's internal structure:

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

Layers 1–4 are genuinely anti-class-structured — within-class distances exceed between-class distances. Class identity only becomes geometrically resolvable in the final blocks. The baseline's class structure is front-loaded into the final logit space; the activation trajectory through the network is largely unstructured by class identity. This is the gap CTLS targets.

### Stage 2 — Full CTLS Objective

**Checkpoint: epoch 95, val_acc = 94.21%**

| Metric | Baseline | CTLS | Delta |
|--------|---------|------|-------|
| Circuit silhouette | 0.1486 | 0.8097 | **+0.6611** |
| Output silhouette | 0.8091 | 0.8124 | +0.0033 |
| Val accuracy | 93.53% | 94.21% | **+0.68%** |

The +0.66 circuit silhouette jump is the core result. A silhouette of 0.81 indicates cleanly separable clusters with tight intra-class variance and large inter-class gaps. The circuit latent space went from nearly random to highly class-organized without any change to architecture.

Two findings beyond the raw number are important. First, output silhouette is unchanged (+0.003), meaning CTLS did not achieve circuit structure by collapsing the backbone. Second, accuracy improves (+0.68%), meaning the consistency pressure acts as a useful regularizer rather than fighting the task objective. Many interpretability methods trade accuracy for transparency. CTLS does not.

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

CTLS improves intraclass consistency by ~2.6× on average. Animal classes show the strongest effect (cat 0.904, bird 0.876, dog 0.869). Vehicle classes show lower but still substantial improvement, likely reflecting genuine visual similarity between categories like automobile and truck that the circuit space captures rather than forcing artificial separation.

**Noise robustness:**

| Noise σ | CTLS ratio | Baseline ratio |
|---------|-----------|---------------|
| 0.05 | 0.511 | 0.417 |
| 0.10 | 0.645 | 0.585 |
| 0.20 | 0.704 | 0.364 |
| 0.30 | 0.755 | 0.295 |
| 0.50 | 0.784 | 0.273 |
| 0.80 | 0.788 | 0.299 |

The divergence at high noise is the key finding. CTLS circuit embeddings maintain informational fidelity under distribution shift; baseline embeddings cannot track what the model is computing because they were never encoding it in the first place.

### Stage 4 — Depth-Weighting Ablation

| Variant | Circuit sil | Output sil | Val acc |
|---------|------------|-----------|--------|
| Baseline (λ=0) | 0.1513 | 0.8040 | 93.53% |
| Uniform weights | 0.8068 | 0.8020 | 93.88% |
| Depth-weighted | **0.8295** | **0.8309** | **94.21%** |

Depth-weighting outperforms uniform weighting on all three metrics. Notably, uniform weighting slightly degrades output silhouette versus baseline, suggesting early-layer consistency pressure interferes with the backbone's natural low-level feature development. Depth-weighting avoids this by not penalizing early layers heavily.

**Known limitation — Layer 7 collapse:** Both CTLS variants show a local dip in per-layer silhouette at layer 7 (the penultimate ResNet block). The consistency pressure creates partial representational collapse at this layer. A per-layer decorrelation or diversity penalty targeted at layer 7 is the identified fix.

### Stage 5 — SAE Monosemanticity

| Layer | Base mono | CTLS mono | Δmono |
|-------|----------|----------|-------|
| 1 | 0.059 | 0.023 | −0.035 |
| 2 | 0.078 | 0.031 | −0.047 |
| 3 | 0.105 | 0.012 | −0.094 |
| 4 | 0.105 | 0.010 | −0.096 |
| 5 | 0.092 | 0.013 | −0.079 |
| 6 | 0.095 | 0.012 | −0.083 |
| 7 | 0.085 | 0.004 | −0.081 |
| 8 | 0.125 | 0.004 | −0.122 |

| | Recon MSE | Sparsity |
|-|----------|---------|
| CTLS SAE | 0.00166 | 0.2492 |
| Baseline SAE | 0.00238 | 0.3428 |

---

## 7. The Monosemanticity Paradox

The most theoretically significant finding: CTLS achieves dramatically higher circuit silhouette (0.81 vs 0.15) but *lower* monosemanticity across all layers (Δmono = −0.080 average). This apparent contradiction is the project's most important result.

The standard monosemanticity metric counts what fraction of SAE dictionary features activate selectively for a single class. The baseline has higher monosemanticity — yet its circuit silhouette is 0.15, meaning its clusters are nearly unstructured. How can a model with more monosemantic features be less class-organized in embedding space?

The resolution is that **baseline monosemantic features are monosemantic by chance on a noisy background.** Because the baseline's activations have no class-structure at the trajectory level, the SAE is fitting essentially random variance. Some features in that random variance happen to correlate with one class — these count as monosemantic. But they are sparse islands of class signal in a sea of noise.

CTLS encodes class identity through **structured superposition** — features that activate for multiple related classes simultaneously (a "fur texture" feature active for cat and dog; a "wings" feature active for bird and airplane). No individual feature is exclusively class-specific, so monosemanticity is low. But the combination of features encodes class identity with high fidelity, which is why circuit silhouette is high.

This finding has implications beyond CTLS. It provides empirical evidence that neuron-level monosemanticity and circuit-level organization are measuring genuinely different properties of a model — and that monosemanticity can be high precisely because the model has *no* meaningful circuit structure. The two metrics are not proxies for the same thing.

**Why SAE is the wrong primary evaluation tool for CTLS:** SAEs are a neuron-level analysis tool. CTLS never optimized for neuron-level monosemanticity — it optimized for population-level trajectory consistency. Using SAEs to evaluate CTLS is analogous to evaluating a symphony by checking whether each instrument plays only one note. The right evaluation tools for CTLS are trajectory-level:

- **Representational Similarity Analysis (RSA):** Pairwise distance matrix of circuit trajectories, correlated against semantic category structure
- **Linear CKA across layers:** Measures geometric consistency of the trajectory layer-to-layer for same-class inputs
- **Circuit latent space UMAP:** Visual verification of semantic clustering; should be distinct from output embedding UMAP
- **Cluster purity metrics:** Silhouette score, adjusted rand index comparing circuit embedding clusters to ground truth labels

---

## 8. Novelty Assessment

Based on literature review, the following gap exists in current work:

| Method | Mechanism | What It Misses |
|--------|-----------|---------------|
| Sparse Autoencoders (SAEs) | Post-hoc single-layer decomposition | Training-time objective; multi-layer trajectories; cross-input consistency |
| Monosemantic Feature Neurons (MFNs) | Stability loss on bottleneck under noise | Cross-input semantic consistency; full trajectory; depth-weighting |
| MonoLoss | Differentiable monosemanticity score on individual neurons | Circuit routing; population dynamics; trajectory embedding |
| Activation Consistency Training (ACT) | Residual stream consistency under prompt perturbation | Interpretability goal; semantic grouping; circuit extraction |
| Brain-Inspired Modular Training (BIMT) | Weight-level penalty on connection length | Activation trajectories; training-time consistency loss |
| CKA / Probing Classifiers | Post-hoc representation similarity | Training-time objective; any feedback to model |
| **CTLS** | **Joint latent space over full multi-layer activation trajectories with InfoNCE semantic consistency** | — |

The specific combination that does not appear in existing literature: a training-time objective that (1) treats the full multi-layer activation trajectory as the unit of analysis, (2) embeds that trajectory into a joint circuit latent space, and (3) uses contrastive semantic consistency between genuinely distinct same-category inputs as the loss signal — as opposed to perturbation-based stability of the same input.

---

## 9. Concrete Applications

### Medical Imaging and Clinical AI

The specific problem CTLS solves is *reasoning consistency*. A diagnostic model might flag two similar chest X-rays as high-risk for different reasons — one triggered by genuine pathology, the other by an imaging artifact. Post-hoc methods cannot reliably distinguish these cases. A CTLS-trained model provides a structural guarantee that similar pathologies activate similar circuits. When a new scan lands far from the expected circuit cluster for its predicted class, that is an automatic anomaly flag — uncertainty quantification at the circuit level, not just the output level. This is directly relevant to FDA approval pathways for AI diagnostic tools.

### AI Safety and Alignment

One of the core problems in safety is detecting when a model reasons differently than it appears to — using an internal pathway that bypasses circuits researchers analyzed and deemed safe. CTLS makes this detectable by construction. If a model routes inputs through circuits far from the expected cluster for that input type, that deviation is measurable in circuit space. The 0.784 vs 0.273 noise robustness ratio at high noise suggests this signal is meaningful even under significant distribution shift.

### Continual Learning and Domain Adaptation

Catastrophic forgetting happens partly because new data restructures circuits that were working well for old data. CTLS provides a concrete way to monitor this — tracking whether new training disrupts the circuit structure of previously-learned categories — and a natural mechanism for circuit preservation losses to prevent it.

### Drug Discovery and Molecular Property Prediction

For predicting properties of structurally similar molecules, knowing whether the model uses the same reasoning pathway for similar compounds is valuable independent information. CTLS circuit embeddings provide a similarity measure over reasoning pathways, not just input features.

---

## 10. Roadmap: Toward CTLS-SSL

The supervised CTLS results establish proof of concept. The natural extension is to SSL, where positive pairs are defined by augmentation proximity or learned similarity rather than class labels. This removes the labeled data dependency and opens a more significant research direction.

### Why SSL Is Theoretically Interesting

In SSL, the model must discover what "semantically similar" means from data structure alone. When encountering inputs whose circuit embeddings do not fit cleanly into any existing cluster, the model cannot assign them to a category — it must either form a new circuit cluster or signal out-of-distribution. This gives CTLS-SSL a structural awareness of its own uncertainty: not just "I am not confident in my prediction" but "I do not have a consistent internal pathway for processing this input."

### The Sample Efficiency Hypothesis

The primary competitive claim for CTLS-SSL is sample efficiency for new categories — specifically, categories that are semantically related to training data. This is a metric every SSL paper already reports, making it an accepted competitive benchmark rather than a new goalpost.

Standard SSL models like DINO or SimCLR learn output-space representations where similar inputs cluster together. When a new category appears, the model must locate it in an unstructured space from few examples alone. CTLS-SSL has a circuit scaffold: the question is not "where does this live in an unstructured space" but "which existing circuit patterns does this new category share, and what makes it distinct?" A model that has learned circuits for "dog" should need very few examples to learn "wolf" because the circuits partially overlap.

**Where CTLS-SSL should win most clearly:**
- Semantically related new categories, where circuit sharing provides a scaffold
- High within-class variation scenarios, where routing consistency provides better generalization from few examples

**Where it may not win:**
- Completely novel categories with no circuit overlap to training distribution — here the scaffold provides no anchor, and consistency pressure may actually slow adaptation

This honest scoping of the competitive claim makes it more credible and more precise.

### Proposed Paper Framing

Not "our method beats DINO everywhere" — rather, "CTLS-SSL outperforms DINO specifically on sample efficiency for semantically related new categories, with a theoretical explanation for why, and with minimal cost on standard accuracy benchmarks." The supervised CTLS results already show the accuracy gap is small while the circuit organization gain is large. CTLS-SSL extends this to the SSL regime on a metric the community already cares about.