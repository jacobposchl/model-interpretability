# Phase 1: Meta-Encoder Validation — Detailed Technical Synthesis

## 1. Overview and Motivation

Phase 1 is a self-supervised representation learning problem. The goal is to train a meta-encoder whose learned space — referred to as **circuit space** — faithfully reflects the internal computational structure of a fixed backbone network, specifically the multi-layer trajectory that any given input traces through the network. This is not about predicting class labels, nor about learning representations that cluster by output behavior. The objective is strictly about encoding *how* a network processes information, layer by layer, regardless of what it ultimately decides.

The central claim motivating this framework is that neural networks do not process every input differently. Instead, they rely on recurring computational pathways — stable, reusable patterns of activation that appear across many inputs, often spanning multiple contiguous layers. These are what we call **circuits**. If the meta-encoder can learn a space that preserves the geometry of these pathways, then the structure of circuit space will itself become interpretable: similar positions in the space correspond to inputs that were processed similarly by the backbone, and the layer-by-layer geometry of the space reveals *where* in the network that similarity occurs.

This phase makes no attempt to modify or optimize the backbone. The backbone is fixed throughout. The meta-encoder is trained to observe and represent; it is a reader, not a writer. Validation of this phase therefore means demonstrating that the reader has learned something real — that circuit space genuinely encodes multi-layer trajectory structure, that it can be used to discover stable recurring patterns, and that those patterns correspond to identifiable computational features of the network.

---

## 2. Trajectory Extraction from the Backbone

### 2.1 Backbone and Layer Outputs

A standard convolutional backbone (ResNet18 in the primary configuration) processes an input `x` and produces a sequence of intermediate outputs across its `L` blocks:

```
T(x) = (h_1, h_2, ..., h_L)
```

where `h_l` is the post-activation output of block `l`. Each `h_l` is globally average-pooled (GAP) to shape `[B, D_l]`, collapsing spatial dimensions and retaining only the channel-wise activation summary.

**Assumption to validate:** GAP discards spatial structure entirely. Two inputs could produce very different spatial activation maps at a given layer but arrive at nearly identical global averages. This framework therefore treats *what* the layer responds to as more informative than *where* it responds, and accepts that any circuit structure expressed in spatial organization will be invisible to the meta-encoder. This is a deliberate simplifying assumption justified by its computational tractability and alignment with classification-oriented backbones, but it warrants empirical testing. A secondary experiment comparing GAP against spatial max-pooling or top-k activation pooling is planned (see Section 7, Experiment 4).

### 2.2 L2 Normalization of Layer Outputs

Before any projection, each `h_l` is L2-normalized:

```
h_l_hat = h_l / ||h_l||_2
```

This step is essential. Without it, later layers, which tend to produce significantly higher activation norms than early layers, would dominate every distance metric in the representation space. The geometry of circuit space would collapse toward final-layer structure regardless of what the meta-encoder learns. By normalizing each layer's output independently, we remove magnitude differences across depth and ensure that every layer contributes comparably to the trajectory representation.

### 2.3 The Alignment Profile

For any two inputs `a` and `b`, the **alignment profile** is defined as:

```
P(a, b) = [s_1(a,b), ..., s_L(a,b)] ∈ R^L
```

where each element is the per-layer cosine similarity between the normalized activations of the two inputs:

```
s_l(a, b) = cos(h_l_hat^a, h_l_hat^b)
```

The profile is the central object of the method. It encodes not whether two inputs are globally similar, but the precise layer-by-layer map of where the network agrees or diverges in how it processes them. A profile that is uniformly high across all layers indicates two inputs share the same computation end-to-end. A profile that is high at early layers and low at later layers suggests a shared low-level circuit that diverges into distinct high-level representations. A profile that is high only in a narrow window of contiguous layers is the clearest signal of a localized circuit — a stable, recurring computational pattern confined to a specific depth range.

All backbone activations used to compute the profile are treated with stop-gradients. The profile is the ground truth signal; the backbone's structure is not to be modified.

---

## 3. Meta-Encoder Architecture

### 3.1 Per-Layer Projectors

Each normalized layer output `h_l_hat` is projected into a common `d`-dimensional space via a dedicated linear projector with LayerNorm and GELU activation:

```
p_l = LayerNorm(GELU(W_l · h_l_hat))     shape: [B, d]
```

Each layer has its own projector `W_l`, so the architecture is capable of learning layer-specific transformations. Because all projectors map to the same dimensionality `d`, the architecture is agnostic to variation in layer widths across the backbone — `D_l` can be different for every layer, but `p_l` is always `d`-dimensional. This is what allows the meta-encoder to be backbone-agnostic in principle, though Phase 1 is validated on ResNet18.

The choice of LayerNorm and GELU here is a pragmatic one. It would be valuable to experiment with alternative projector designs, such as removing the nonlinearity entirely (pure linear projection), or using a small two-layer MLP per projector. The current design represents a middle ground between expressiveness and simplicity.

### 3.2 Transformer with Rotary Position Embeddings

The projected layer tokens `p_1, ..., p_L` are passed through a transformer encoder, where position information is injected via **Rotary Position Embeddings (RoPE)**:

```
[p_1_rope, ..., p_L_rope] → transformer → [z_1, ..., z_L]
```

RoPE encodes layer depth by rotating the query and key vectors of the attention mechanism by an angle proportional to each token's layer index. The key property of RoPE is that the dot product between any two layer tokens — which drives attention weighting — naturally decays as a function of their relative depth distance. This gives the transformer an inductive bias that makes it easier to represent relationships between nearby layers than between distant ones, without imposing a hard locality constraint. The transformer can still attend across arbitrary layer distances when the content warrants it; RoPE simply makes the prior over attention patterns favor contiguity, which aligns with the intuition that circuits tend to span contiguous layer ranges rather than arbitrary, non-local combinations.

This is more principled than sinusoidal positional embeddings for this problem. Sinusoidal encodings inject position information additively, meaning the position signal mixes with the token content. RoPE injects position multiplicatively into the similarity computation itself, meaning the geometric relationship between two positions is expressed directly in how much those tokens attend to each other, which is the relationship we care about.

The transformer produces per-layer outputs `z_1, ..., z_L`, which are the only quantities exposed from the meta-encoder's forward pass. These are the representations used for all downstream circuit analysis.

---

## 4. Training Objective

The full training loss consists of two terms:

```
L = L_info + λ · L_geometry
```

Both terms are necessary and serve distinct purposes. The fidelity term ensures representations are *informative* about the profile structure. The geometry term ensures the learned space is *organized* such that downstream clustering recovers real sub-trajectory circuits. Fidelity alone does not enforce useful geometric structure. Geometry alone does not constrain representations to encode the per-layer internal profile. The interplay between them is what makes circuit space both interpretable and usable.

### 4.1 Fidelity Term: L_info

The fidelity term trains each per-layer representation to contain sufficient information to reconstruct the corresponding entry in the alignment profile:

```
L_info = (1/L) Σ_l || MLP(z_l^a ⊙ z_l^b) - s_l(a,b) ||²
```

The key design decision here is the use of the **element-wise product** `z_l^a ⊙ z_l^b` to combine the two per-layer representations before passing them to the MLP regressor.

This choice is mathematically justified as follows. The ground truth `s_l(a, b)` is a cosine similarity — a scalar that is symmetric by definition (swapping `a` and `b` produces the same value). The combination function must therefore also be symmetric. Simple concatenation `[z_l^a, z_l^b]` is asymmetric: an MLP applied to a concatenated pair can learn to treat the first argument differently from the second, introducing an inductive bias that contradicts the symmetry of the target. The element-wise product is symmetric by construction (swapping `a` and `b` gives the same vector), and it preserves per-dimension co-activation information — each dimension `i` of `z_l^a ⊙ z_l^b` encodes how much both representations simultaneously activate on dimension `i`. The MLP can then learn which dimensions are most predictive of backbone cosine similarity.

Note that if both `z_l^a` and `z_l^b` are L2-normalized (as they will be for the geometry term), summing all entries of their element-wise product gives exactly their dot product, which equals their cosine similarity. The element-wise product strictly dominates the dot product in expressiveness because the MLP sees the full per-dimension structure rather than a pre-collapsed scalar. The difference encoding `|z_l^a - z_l^b|` is a symmetric alternative, but it is geometrically less natural here because the target is a similarity rather than a distance — the difference encoding requires the MLP to learn the additional inversion that large differences correspond to low similarities.

### 4.2 Geometry Term: L_geometry

The geometry term enforces that inputs with similar alignment profiles are geometrically close in `z` space, using a soft contrastive objective:

```
L_geometry = (1/L) Σ_l [ -Σ_{a,b} P̄_l(a,b) · log( exp(sim(z_l^a, z_l^b) / τ) / Σ_c exp(sim(z_l^a, z_l^c) / τ) ) ]
```

where `P̄_l(a,b)` is `s_l(a,b)` normalized across all pairs in the batch to sum to 1, acting as a soft target distribution over positives for the contrastive loss, and `τ` is a temperature parameter.

All `z` vectors are L2-normalized onto the unit hypersphere prior to similarity computation in this term. This provides implicit geometric repulsion: similar pairs are pulled together on the sphere, and all other pairs are pushed apart by the bounded structure of the sphere itself, without any explicit repulsion term.

The temperature `τ` controls how sharply the contrastive distribution focuses on high-similarity pairs. Low `τ` concentrates gradients onto the pairs that are already most similar (forcing tighter clustering), while high `τ` spreads gradient signal more evenly across the batch.

**Potential failure mode to monitor:** If a batch happens to contain inputs with uniformly similar alignment profiles — which can occur early in training or with poorly shuffled data — the normalized distribution `P̄_l(a,b)` becomes nearly uniform across all pairs. In this regime, every pair is treated as an equally strong positive, the contrastive loss receives diffuse gradients, and the geometry term stops contributing meaningful signal. A diagnostic for this is to log the effective entropy of `P̄_l(a,b)` across training batches. If entropy consistently approaches its maximum, it may be necessary to introduce batch-level stratification (ensuring each batch contains diverse trajectory profiles) or to revise the normalization strategy.

---

## 5. Circuit Discovery Procedure

### 5.1 The Raw Profile Vector as the Foundation

After training, the meta-encoder is used offline for circuit discovery. The procedure begins by passing `N` inputs through both the backbone and the trained meta-encoder, collecting the raw alignment profiles `P(a, b)` for all input pairs and the per-layer z-representations `z_l` for each individual input.

The raw alignment profile vector is the foundational object for all discovery. For any pair `(a, b)`:

```
P(a, b) = [s_1(a,b), s_2(a,b), ..., s_L(a,b)] ∈ R^L
```

Each entry is the per-layer cosine similarity computed during training. Crucially, this vector is **not** collapsed, globally normalized, or summarized before discovery. A pair whose network processing agrees at layers 3–5 and 7–8 but diverges everywhere else produces a profile like `[low, low, high, high, high, low, high, high, low]`. This raw vector encodes all of that information simultaneously and faithfully.

It is essential to understand why a global softmax across all L layers must *not* be applied to this vector before clustering. Applying a global softmax normalizes all values relative to each other across the entire depth of the network, turning the profile into a single probability distribution over layers. This collapses multi-circuit pairs: a pair whose profile is `[low, low, high, high, high, low, high, high, low]` gets converted into a blurred, two-peaked distribution where neither peak is cleanly preserved as a distinct object. More critically, this pair and a second pair whose profile is `[low, low, high, high, high, low, low, low, low]` — sharing only the 3–5 circuit — would produce very different global distributions and would be separated by clustering, even though they share a real circuit at layers 3–5. The global softmax discards multi-circuit structure by forcing each pair to be characterized as one thing.

The correct approach preserves the raw profile and performs all analysis at the level of individual spans, described below.

### 5.2 Span-Centric Clustering: One Pass Per Candidate Span

A circuit is defined as a contiguous span of layers `[l_start, l_end]`. Rather than inferring which span a pair belongs to from a globally normalized profile, the discovery procedure enumerates all possible contiguous spans and asks, for each span independently: which pairs have high and consistent similarity *within this span, regardless of what they do outside it?* This is the critical structural shift. A pair can simultaneously belong to multiple span-circuits, because it is evaluated independently for each span. This directly mirrors the multi-circuit reality of network computation.

The full set of candidate spans is `{[l_start, l_end] : 1 ≤ l_start ≤ l_end ≤ L}`. For a backbone with L = 8 layers (ResNet18), this produces `L(L+1)/2 = 36` candidate spans. This is a fixed, small, tractable set — enumeration is not a computational concern.

For each candidate span `S = [l_start, l_end]`, the procedure is as follows.

**Step 1 — Extract the span sub-vector.** For every pair `(a, b)`, extract the contiguous slice of the raw profile corresponding to this span:

```
P_S(a, b) = [s_{l_start}(a,b), s_{l_start+1}(a,b), ..., s_{l_end}(a,b)] ∈ R^{|S|}
```

where `|S| = l_end - l_start + 1` is the number of layers in the span. This sub-vector describes only how the two inputs relate to each other at the layers within this span, with no reference to what happens outside it.

**Step 2 — Apply within-span temperature sharpening.** A temperature-scaled softmax is applied *within the span sub-vector only*, not across the full profile:

```
Q_S^l(a, b) = exp(s_l(a,b) / τ) / Σ_{l' ∈ S} exp(s_{l'}(a,b) / τ)
```

This sharpened sub-vector `Q_S(a, b) ∈ R^{|S|}` serves as a within-span distribution that highlights which layers inside this particular span are most active for this pair. The temperature `τ` controls the sharpness of this weighting: at low `τ`, the distribution concentrates on the single highest-similarity layer within the span; at high `τ`, it spreads weight more evenly across all span layers. Crucially, the temperature is applied only within the span, so a pair with high similarity in just two of the span's five layers will correctly produce a peaked distribution at those two layers without being influenced by what happens at layers outside the span.

**Step 3 — Cluster pairs by their within-span sharpened sub-vector.** Run HDBSCAN on the set of all `N(N-1)/2` sharpened sub-vectors `{Q_S(a, b)}` for this span. Each resulting cluster is a group of input pairs that agree in the same way within this span — they are similar at the same sub-set of layers inside `[l_start, l_end]`, with similar sharpness.

**Step 4 — Apply the canonicality criterion.** A cluster is a **canonical circuit candidate** for span `S` if it contains more than a minimum fraction of all input pairs (e.g., >1% of all pairs). This threshold ensures the pattern is a stable, recurring computational primitive rather than an artifact of a small number of similar inputs. Clusters below this threshold are discarded for this span.

This procedure is repeated independently for every candidate span. The total set of canonical circuit candidates across all spans constitutes the discovered circuit vocabulary for this backbone.

### 5.3 Multi-Circuit Membership is Expected and Correct

Because span-centric clustering is performed independently for each span, a single input pair will naturally appear in multiple canonical circuit clusters — one for each span over which it has high, recurring similarity. This is the correct behavior and should not be treated as a problem or a sign of redundancy. It directly reflects that a given pair of inputs may share a low-level texture circuit at layers 2–4, diverge at the mid-level, and then share a shape-recognition circuit again at layers 6–8. These are two distinct canonical circuits, and the pair belongs to both of them. No single global clustering could recover this structure.

This also means that the *frequency profile of a given input* — how many and which canonical circuits it participates in across all spans — is itself a rich descriptor of how that input is processed by the backbone. Inputs that participate in many canonical circuits are being processed through a large number of stable, shared computational patterns. Inputs that participate in few are being processed via idiosyncratic trajectories. This frequency profile is a secondary analysis worth pursuing after the primary circuit discovery is validated.

### 5.4 Circuit Representation in Z-Space

Once canonical circuit candidates have been identified for a given span `S`, the z-representations are used to characterize each circuit geometrically. For all input pairs belonging to a given span-cluster, and for all layers `l ∈ S`, the per-layer z-vectors `z_l` for each input in the pair are collected. This gives a tensor of shape `[|cluster_inputs|, |S|, d]`, where `|cluster_inputs|` is the number of unique inputs appearing across all pairs in the cluster. The centroid of this tensor, averaged over the input dimension and flattened over the span dimension, gives the **canonical circuit prototype** — a representative point in circuit space for this particular computational pattern at this particular span.

The spread (variance) of the cluster around its prototype quantifies how tight or diffuse the circuit is. A tight cluster with low variance indicates a highly stereotyped computation — the network processes all inputs in this cluster very similarly across the span. A diffuse cluster indicates a broader family of computations that share some structural similarity but with meaningful variation.

This prototype structure is the foundation of the eventual circuit vocabulary: each canonical circuit is a `(span, prototype)` pair, and a new input can be mapped into circuit space and compared against all prototypes to identify which circuits it activates, at which spans, and how strongly. Critically, because membership is span-specific, a new input can be assigned to zero, one, or many canonical circuits simultaneously, which is the expressive multi-circuit characterization that the global profile approach could not support.

---

## 6. Success Criteria

Phase 1 validation is considered successful if and only if all of the following quantitative criteria are met. Visual inspection of circuit clusters is used as a secondary sanity check but is not sufficient on its own.

**Criterion 1 — Profile Reconstruction Accuracy:** After training, the average mean-squared error of the MLP regressor on held-out inputs must be significantly below the baseline of always predicting the mean profile value. A target reconstruction R² of at least 0.7 on held-out data is the primary bar. This directly measures whether `L_info` is doing its job: encoding sufficient profile information into the z-representations to recover per-layer backbone similarity.

**Criterion 2 — Geometric Consistency:** The geometry term must have produced a well-organized space. Concretely: for any two inputs `a` and `b` with high true profile similarity at layer `l`, their z-vectors `z_l^a` and `z_l^b` should be geometrically close. This is measured as Spearman correlation between the pairwise true profile similarity `s_l(a, b)` and the cosine similarity of the learned representations `sim(z_l^a, z_l^b)`, computed per layer on a held-out set. A Spearman ρ above 0.5 per layer, and above 0.65 averaged across all layers, constitutes a pass.

**Criterion 3 — Within-Span Similarity Elevation:** For each canonical circuit candidate at span `S`, the mean within-span similarity across all pairs in the cluster must be significantly higher than the population mean within-span similarity for that same span computed across all pairs. Concretely, the cluster mean must exceed the population mean by at least one population standard deviation. This verifies that the clustering is recovering genuinely high-similarity sub-trajectories, not arbitrary groupings that happen to meet the size threshold by chance. Because the span is fixed before clustering in the span-centric procedure, there is no span endpoint variance to measure — the coherence criterion is about similarity elevation within the span, not span localization.

**Criterion 4 — Circuit Diversity:** The discovered canonical circuits must not all collapse to the same span. The set of discovered circuit prototypes must span at least 60% of the total layer range `[1, L]`, meaning the framework must discover circuits at multiple depth levels (early, mid, late). Collapse to a single depth region would indicate that the backbone's trajectory structure is not meaningfully diverse, or that the meta-encoder has failed to differentiate depth.

**Criterion 5 — Class Purity Distribution is Bimodal or Mixed:** Among all canonical circuit candidates with >1% of pairs, the distribution of class purity scores must not be uniformly high (all circuits class-pure) or uniformly low (no class-specific circuits). A healthy distribution should show both class-agnostic circuits (purity < 0.3) and class-specific circuits (purity > 0.7), indicating that circuit space is organizing computation at multiple levels of semantic abstraction.

---

## 7. Validation Experiments

The following experiments are designed to be run in sequence, where each one builds on the results of the previous. Together, they constitute a complete validation procedure for Phase 1.

### Experiment 1: Profile Reconstruction Fidelity (Core Validation)

**Purpose:** Verify that the fidelity term `L_info` is working — that the per-layer z-representations encode sufficient information to reconstruct per-layer profile similarities.

**Setup:** Train the full meta-encoder on ResNet18 representations from CIFAR-10. After training, freeze the meta-encoder. Draw a held-out set of 2,000 inputs, compute all pairwise profiles `P(a, b)`, and run the MLP regressor on `z_l^a ⊙ z_l^b` for every pair and every layer. Compute R² and mean absolute error between predicted and true profile values.

**Ablations to run alongside:** Train a version with `L_info` only (no geometry term). Train a version with `L_geometry` only (no fidelity term). This directly validates the claim in the design that both terms are necessary — fidelity alone should produce high reconstruction accuracy but poor geometric organization, while geometry alone should produce organized space but poor reconstruction.

**Expected result:** Full model achieves R² ≥ 0.7 on held-out profiles. Geometry-only ablation produces R² significantly below this. Fidelity-only ablation produces R² similar to full model but fails Criterion 2 in Experiment 2.

**Failure mode:** If R² is low for the full model, the most likely causes are: the MLP regressor is too shallow; the projectors are not learning layer-discriminative representations; or the backbone's profile structure has too little variance to learn from (all profiles similar). Diagnostic: compute and plot the true profile variance across the held-out set — if it's near zero, the backbone is not being given diverse enough inputs.

---

### Experiment 2: Geometric Consistency (Z-Space Structure)

**Purpose:** Verify that the geometry term `L_geometry` has induced the correct structure in z-space — that inputs with similar profiles are geometrically close at the corresponding layers.

**Setup:** On the same held-out set from Experiment 1, compute Spearman correlation between true pairwise profile similarity `s_l(a, b)` and learned z-space similarity `sim(z_l^a, z_l^b)` for each layer `l` separately. Plot this correlation as a function of layer depth. Additionally, apply UMAP to the z-vectors at each layer separately and visualize how well the true profile structure organizes the low-dimensional embedding.

**Expected result:** Spearman ρ > 0.5 at every layer, > 0.65 averaged across layers. UMAP plots should show smooth, interpretable organization — inputs with high profile similarity at a given layer should cluster visually.

**Failure mode:** If correlation is high at the final layer but low at early layers, the meta-encoder has collapsed to final-layer structure, suggesting the L2 normalization of activations is not sufficient to counteract backbone depth bias. Consider strengthening normalization or adding a layer-depth penalty to the loss. If correlation is uniformly low, the geometry term is not producing gradient signal — monitor the entropy of `P̄_l(a,b)` as described in Section 4.2.

---

### Experiment 3: Circuit Discovery and Span Validation

**Purpose:** Verify that the span-centric discovery procedure recovers coherent, contiguous, canonical circuits — the primary empirical claim of the entire framework — and that the procedure correctly handles inputs that simultaneously participate in multiple circuits at different depth ranges.

**Setup:** Pass N = 10,000 inputs from CIFAR-10 through the trained backbone and meta-encoder. Collect all pairwise raw profile vectors `P(a, b) ∈ R^L`. Enumerate all 36 candidate spans for ResNet18. For each span `S = [l_start, l_end]`, extract the within-span sub-vector `P_S(a, b)` for every pair, apply within-span temperature sharpening with a sweep of `τ ∈ {0.1, 0.5, 1.0, 2.0}`, and cluster the resulting sharpened sub-vectors using HDBSCAN. Apply the >1% size criterion to identify canonical circuit candidates per span. For each candidate, record: the span `[l_start, l_end]`; the cluster size (number of pairs); the mean and variance of within-span similarity values across all pairs in the cluster; and the class purity of the unique inputs participating in pairs in that cluster.

**Multi-circuit membership check:** For a randomly sampled subset of 500 input pairs, record how many distinct canonical circuit clusters each pair appears in across all spans. Plot the distribution of per-pair circuit membership counts. If the framework is working, this distribution should be multimodal — some pairs participate in one or two circuits, others in several — reflecting genuine variation in how much computation inputs share. A distribution concentrated at exactly 1 would suggest the span-centric procedure is collapsing to single-circuit behavior, which would indicate the backbone's trajectory structure is not as compositional as expected. A distribution concentrated at the maximum would suggest the canonicality threshold is too loose.

**Shared-circuit coherence check:** Identify at least one pair of input pairs `{(a,b), (c,d)}` where `(a,b)` has a profile like `[low, low, high, high, high, low, high, high, low]` and `(c,d)` has a profile like `[low, low, high, high, high, low, low, low, low]`. Verify explicitly that they appear in the same canonical circuit cluster for span `[3, 5]` and in different clusters (or no cluster at all) for span `[7, 8]`. This is the key correctness check that validates the span-centric approach over global profile clustering — it should be reported as an explicit unit test result.

**Visualizations to produce:** For each canonical circuit candidate, display a random sample of 16 input images from the cluster. Annotate each image with its raw profile vector, highlighting the values within the circuit's span. If the framework is working, images in low-purity clusters should share visually identifiable low-level features (texture, edge orientation, spatial frequency) at the span's depth level, while images in high-purity clusters should share semantic content. The visual inspection is a sanity check, not a primary criterion, but a total failure of visual coherence is grounds for further debugging.

**Expected result:** Discovery of at least 5 canonical circuit candidates distributed across different spans, collectively covering at least 60% of total network depth (Criterion 4). Within-cluster similarity values should be meaningfully higher than the population mean for that span (the circuit is genuinely capturing high-similarity pairs, not random ones). Multi-circuit membership distribution should be non-degenerate. The shared-circuit coherence check should pass explicitly. A bimodal or mixed class purity distribution satisfying Criterion 5.

**Failure mode:** If all discovered circuits are at the final 1–2 layers, circuit space has collapsed to final-layer structure. If no span produces clusters meeting the >1% size criterion, the backbone's trajectory structure may not be sufficiently recurring at any depth for this dataset — consider whether ResNet18 on CIFAR-10 has enough input diversity and rerun with ImageNet. If the shared-circuit coherence check fails — meaning `(a,b)` and `(c,d)` with matching 3–5 profiles are not co-clustered at span `[3,5]` — there is a bug in the span sub-vector extraction or clustering pipeline that must be resolved before any other results are interpreted.

---

### Experiment 4: Pooling Strategy Ablation

**Purpose:** Test the GAP assumption (Section 2.1) — specifically, whether discarding spatial structure by global average pooling causes meaningful loss of circuit information.

**Setup:** Train three versions of the meta-encoder on ResNet18 with identical hyperparameters, differing only in how `h_l` is computed before projection: (a) global average pooling (baseline), (b) global max pooling, (c) top-k activation pooling where k is set to retain the top 10% of activations by magnitude. Evaluate all three on Criteria 1 through 5.

**Expected result:** Quantitative differences between pooling strategies should be modest if the baseline assumption holds. Qualitatively, max pooling and top-k pooling may produce circuit clusters whose member images share more localized visual features (since they preserve "where" information partially), while GAP clusters share more diffuse, global features. This experiment is expected to produce a recommendation for future phases rather than a definitive answer in Phase 1.

---

### Experiment 5: Temperature Sensitivity Analysis

**Purpose:** Understand how sensitive the discovered circuit structure is to the temperature hyperparameter `τ` — which now appears in two distinct places with different roles — and identify good default values for each.

The first role is `τ_geometry`, used in the geometry loss term `L_geometry` during meta-encoder training. This controls how sharply the contrastive objective clusters similar-profile inputs in z-space. The second role is `τ_discovery`, used during the within-span sharpening step of the circuit discovery procedure. This controls how much weight is concentrated on the highest-similarity layers within each span's sub-vector when clustering pairs into circuit candidates. These are two independent hyperparameters and must be analyzed jointly, because a model trained with a particular `τ_geometry` may interact differently with different values of `τ_discovery`.

**Setup:** Train the meta-encoder with `τ_geometry ∈ {0.05, 0.1, 0.5, 1.0}`. For each trained model, run the full span-centric circuit discovery procedure with `τ_discovery ∈ {0.1, 0.5, 1.0, 2.0}`. For each `(τ_geometry, τ_discovery)` combination, measure all five success criteria. Plot a 4×4 heatmap of each criterion score as a function of both temperatures, producing five heatmaps in total.

**Expected result:** There should exist a region of the temperature grid where all criteria are jointly satisfied. Very low `τ_geometry` should produce over-clustered z-space, collapsing inputs together in z-space regardless of trajectory differences. Very high `τ_geometry` should produce diffuse, poorly organized z-space with low geometric consistency. For `τ_discovery` specifically: at low values, the within-span sharpening concentrates weight heavily on the single highest-similarity layer inside each span, causing clusters to form around very narrow, specific activation patterns; at high values, the sharpening spreads weight evenly, making the clustering sensitive to the average similarity across the full span rather than local peaks within it. The optimal `τ_discovery` is expected to sit around 0.5–1.0, where the distribution is sharp enough to differentiate span-internal structure but not so sharp that it ignores meaningful variation across span layers.

---

### Experiment 6: Transfer Across Backbone Depth

**Purpose:** Verify that the meta-encoder architecture is genuinely backbone-agnostic — that the design's claim of handling variable layer widths via uniform projection applies in practice.

**Setup:** Train the meta-encoder on ResNet18 (8 residual blocks). Then train an identical meta-encoder on ResNet34 (16 residual blocks) and ResNet50 (16 bottleneck blocks, wider channels) with no changes to architecture or hyperparameters. Compare success criteria across all three. Optionally, attempt to use a meta-encoder trained on ResNet18 to read a ResNet34 (zero-shot transfer across backbone depth), which would be the strongest possible evidence for architecture-agnosticity.

**Expected result:** All five criteria should be satisfied across all three backbones with the same hyperparameters, demonstrating that the framework is not overfitting to ResNet18's specific structure. The zero-shot transfer experiment is expected to partially succeed — circuit space geometry may transfer even if the specific circuit span positions shift with network depth.

---

### Experiment 7: Dataset Generalization

**Purpose:** Ensure that circuit discovery generalizes beyond CIFAR-10 and does not depend on the particular input distribution used during training.

**Setup:** Train the meta-encoder on CIFAR-10. Then evaluate it without any retraining on inputs from CIFAR-100 and STL-10. For each dataset, run the full circuit discovery procedure and measure all five success criteria.

**Expected result:** Criterion 1 (reconstruction fidelity) should hold approximately, since it measures a property of the backbone's internal structure and not the specific inputs. Criterion 5 (class purity distribution) will shift — CIFAR-100's 100 classes will tend to produce lower class purity for all clusters simply because purity is harder to achieve with more classes. This experiment provides evidence for the generalizability of the framework and informs whether the meta-encoder is encoding genuine computational structure or overfitting to CIFAR-10's specific class distribution.

---

## 8. Noted Assumptions and Open Questions

The following assumptions are made in the current Phase 1 design and are carried forward with the understanding that they warrant empirical validation, either in Phase 1's ablations or in subsequent phases.

**Pooling assumption:** Global average pooling discards spatial structure. Circuits whose computational signature is expressed spatially (e.g., object localization circuits) will not be captured by the current framework. This is flagged but not addressed in Phase 1.

**Layer output as trajectory signal:** The current framework uses post-activation block outputs as the trajectory signal. It would be informative to explore alternatives: the input to each block (pre-activation), the residual delta between block input and output (which isolates what each block contributes), or the gradient magnitude at each layer (which captures sensitivity rather than activation). These represent fundamentally different conceptions of what "information flowing through a circuit" means, and the choice may significantly affect which circuits are discovered.

**Gradient collapse in L_geometry:** Homogeneous batches can cause the soft target distribution `P̄_l(a,b)` to approach uniform, degrading the geometry term's gradient signal. This is monitored via entropy logging but not actively corrected in Phase 1.

**RoPE encodes depth as rotation, not as hard locality:** The transformer with RoPE can still attend across arbitrary layer distances if the learned content warrants it. The inductive bias favors contiguous spans but does not enforce them. Whether this is sufficient to recover genuinely contiguous circuits — rather than diffuse, non-local patterns — is answered empirically by Experiment 3.

**HDBSCAN cluster stability:** HDBSCAN results can vary with minimum cluster size and other hyperparameters. All clustering results should be reported with stability analysis — running the clustering with multiple random seeds and reporting inter-run agreement via Adjusted Rand Index.