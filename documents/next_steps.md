# Next Steps
### CTLS — Planned Experiments and Future Directions

Each entry has a status, a description of what it is, details on how to implement it, and a progress log. Add new ideas at the bottom of the relevant section. Statuses: `not started` · `in progress` · `completed`

---

## Near-Term: Validation Roadmap (Steps 4–6)

These steps follow directly from the validation experiment (Steps 1–3). They must be done in order — each depends on the previous one.

---

### Step 4 — Activation Extraction Strategy Ablation

**Status:** `not started`

**What it is:**
The current implementation extracts globally average-pooled post-block activations as `h_l`. This is a reasonable default but may not be the representation that best tracks causal circuit structure. This step compares alternative extraction strategies against the existing activation-patching ground truth to find the one with highest proxy validity.

**Details:**
Three alternatives to test against the current approach:
- **Pre-nonlinearity activations** — extract before ReLU/GELU. May carry more signed information since the nonlinearity discards negative activations.
- **Spatially-resolved activations** — flatten instead of average-pool, keeping spatial dimensions. Captures *where* in the image the computation happens, not just *what* activates.
- **Gradient-weighted activations** (GradCAM-style) — weight each activation by its gradient w.r.t. the final logit. More directly causal since it measures each activation's contribution to the output.

For each strategy: swap out the extraction hook, recompute `z` using the same trained meta-encoder (no retraining needed), and rerun Spearman ρ against the existing patching ground truth. The patching results do not need to be re-run — they are a property of the model's causal structure, not of how `z` is extracted.

**Expected outcome:** Whichever strategy produces the highest ρ against patching ground truth is the correct extraction method. If all strategies produce similar ρ, global average pooling is defensible as the default.

**Progress:**
- [ ] Implement spatially-resolved extraction hook
- [ ] Implement pre-nonlinearity extraction hook
- [ ] Implement gradient-weighted extraction hook
- [ ] Run Spearman ρ for each strategy vs. existing patching ground truth
- [ ] Document comparison table and select winning strategy

---

### Step 5 — Positive Pair Definition: Class Labels vs. Causal Similarity

**Status:** `not started`

**What it is:**
The current training signal defines positive pairs by class label — two images of the same class are a positive pair. The patching experiment revealed that ~50% of same-class image pairs have CircuitSim ≈ 0, meaning they produce the same predicted class but use completely different internal circuits. Class labels are a noisy proxy for circuit identity. This step tests whether defining positive pairs directly by patching-derived circuit similarity produces a tighter, more valid training signal.

**Details:**
Two alternatives to test against the current class-label positive pairs:

- **Causal similarity positive pairs:** Images with `CircuitSim(x_a, x_b) > threshold` are positives, regardless of class label. This defines "similar inputs" based on what the model is actually doing internally rather than the semantic category they belong to.
- **Feature-level positive pairs:** Positive pairs share a specific visual feature regardless of class (e.g., all "fur texture" images, all "circular object" images). Requires feature annotation or an auxiliary feature model.

For each definition: retrain the meta-encoder with the new positive pair definition (backbone can be frozen or jointly trained), then evaluate: (1) within-same-class ρ against patching, (2) circuit silhouette, (3) whether the resulting circuit clusters cross class boundaries as expected for causal positive pairs.

**Expected outcome:** Causal positive pairs should yield higher within-same-class ρ against patching than class-label positive pairs. They may yield lower between-class separation since causal similarity sometimes crosses class boundaries (e.g., a blurry dog and a blurry cat may use more similar circuits than a close-up dog and a distant dog). The feature-level variant is exploratory.

**Dependency:** Requires the patching harness from Step 2 (already complete). Optionally benefits from the best extraction strategy identified in Step 4.

**Progress:**
- [ ] Define CircuitSim threshold for positive pair selection
- [ ] Implement causal-similarity pair sampler (replaces PairedCIFAR10 class-based sampler)
- [ ] Retrain with causal positive pairs, evaluate ρ and silhouette
- [ ] Compare against class-label baseline
- [ ] Decide on positive pair definition going forward

---

### Step 6 — SSL Extension (CTLS-SSL)

**Status:** `not started`

**What it is:**
The supervised CTLS results establish proof of concept on labeled data. The natural extension removes the label dependency by replacing class-label-defined positive pairs with augmentation-proximity or patching-derived positive pairs. This is where the method becomes competitive with DINO and SimCLR.

**Details:**
The primary competitive hypothesis: **CTLS-SSL outperforms DINO/SimCLR on sample efficiency for semantically related new categories.** Standard SSL models learn output-space representations — when a new category appears at test time, the model must locate it in an unstructured space from few examples alone. CTLS-SSL has a circuit scaffold: the question is not "where does this live in unstructured space" but "which existing circuit patterns does this new category share, and what makes it distinct?" A model that learned circuits for "dog" should need far fewer examples to learn "wolf" because the circuits partially overlap.

Where CTLS-SSL should win most clearly:
- Semantically related new categories where circuit sharing provides a scaffold
- High within-class variation scenarios where routing consistency provides better few-shot generalization

Where it may not win:
- Completely novel categories with no circuit overlap to training distribution — the scaffold provides no anchor

**Implementation approach:**
- SSL positive pairs: two augmented views of the same image (minimal version) or patching-derived pairs (stronger, requires Step 5 infrastructure)
- Evaluation: few-shot accuracy on held-out semantically related vs. unrelated categories; compare against DINO and SimCLR baselines (configs in `configs/ssl/`)
- Primary metric the community already cares about: few-shot accuracy on related new categories

**Dependency:** Steps 3–5 should be complete first. The right extraction strategy and positive pair definition must be known before extending to SSL.

**Progress:**
- [ ] Confirm Step 4 and Step 5 are complete
- [ ] Implement SSL positive pair pipeline (augmentation-based + optional causal-based)
- [ ] Run CTLS-SSL training (see `configs/ssl/ctls_ssl_v2.yaml`)
- [ ] Run DINO and SimCLR baselines for comparison (see `configs/ssl/`)
- [ ] Evaluate few-shot on semantically related new categories
- [ ] Document results and comparison

---

## Near-Term: Additional Ablations and Validation

These can run in parallel with Steps 4–6 — none depend on their completion. Each is a targeted check that validates or challenges a specific design assumption.

---

### Final-Layer Baseline Ablation

**Status:** `not started`

**What it is:**
The entire trajectory apparatus (multi-layer projection, depth-aware combination, meta-encoder) is only justified if it outperforms a much simpler alternative — plain cosine similarity between final-layer activations alone. If the last layer already captures enough circuit structure to achieve comparable proxy ρ against patching, the trajectory machinery isn't adding value.

**Details:**
Compute pairwise cosine similarity using only `h_8` (the final ResNet block output) and measure Spearman ρ against the existing patching ground truth. No retraining needed — the patching results are fixed.

**Expected outcome:** If the full trajectory proxy substantially outperforms (ρ gap > 0.1), the multi-layer approach is justified. If not, the contribution needs to be reframed.

**Progress:**
- [ ] Compute pairwise cosine similarity using h_8 only on the validation set
- [ ] Measure Spearman ρ against existing patching ground truth
- [ ] Compare against trajectory proxy ρ (0.717–0.743) and document the gap

---

### Uniform Weighting Ablation

**Status:** `not started`

**What it is:**
The linear depth ramp (`w_l = l / sum(1..L)`) was motivated by the assumption that later layers encode more semantically consistent representations. The patching validation contradicted this — empirical causal influence is flat across all 8 layers, not increasing with depth. This makes the depth ramp an uncalibrated prior rather than an empirically grounded design choice.

**Details:**
Retrain Option A with uniform weighting (`w_l = 1/L` for all `l`) and compare proxy Spearman ρ against the current linear ramp and the existing patching ground truth.

**Expected outcome:** If uniform weighting achieves equal or better ρ, the depth-weighting rationale needs revision — the ramp is a useful training heuristic, not a reflection of circuit structure. If the ramp clearly wins, that finding itself is worth reporting.

**Progress:**
- [ ] Train weighted_sum variant with uniform weights (add config or small code change)
- [ ] Run proxy validation (Spearman ρ against patching ground truth)
- [ ] Compare: uniform vs linear ramp, document result

---

### CIFAR-100 Replication

**Status:** `not started`

**What it is:**
All results are currently on CIFAR-10 — 10 coarse classes, relatively easy circuit discrimination. CIFAR-100 has 100 fine-grained classes with high within-class variation, making circuit discrimination harder and the ~50% same-class split finding potentially more pronounced. A reviewer will ask whether the proxy holds at scale.

**Details:**
Replicate the full pipeline on CIFAR-100 with ResNet18: unified objective training, activation patching ground truth on a 1000-pair sample, proxy Spearman ρ.

**Expected outcome:** If ρ holds above 0.6, the scale concern is partially addressed. If it degrades significantly, that is an honest and important limitation to report.

**Progress:**
- [ ] Adapt data loader to CIFAR-100 paired sampling
- [ ] Train unified objective on CIFAR-100
- [ ] Run activation patching on 1000-pair sample
- [ ] Compute proxy Spearman ρ, compare against CIFAR-10 results

---

### UMAP Circuit Space Visualization

**Status:** `not started`

**What it is:**
The ~50% same-class split finding is difficult to communicate via tables alone. A UMAP projection of z-space, colored by class, makes the core finding visually immediate — same-class inputs splitting into sub-clusters, cross-class inputs grouping by shared computational routing, a structure that does not map cleanly onto the label taxonomy.

**Details:**
- Side-by-side: baseline z-space UMAP (roughly random) vs. CTLS z-space UMAP (class-organized, visible within-class sub-structure)
- Color by class label; annotate sub-clusters manually or via k-means
- Optional: 3D version with depth axis to animate how trajectories resolve from ambiguous in early layers to class-specific in later layers
- Already buildable from existing z vectors using `evaluation/circuit_viz.py`

**Progress:**
- [ ] Generate UMAP of baseline z-space vs. CTLS z-space (side-by-side)
- [ ] Annotate within-class sub-clusters
- [ ] Identify cross-class groups that cluster together (shared circuit routing)
- [ ] Optionally: build per-layer animation of trajectory resolution

---

## Longer-Term Directions

These are not blocked by Steps 4–6 in terms of conceptual development, but are lower priority. They represent extensions and applications of the validated CTLS framework.

---

### Hierarchical Semantic Consistency Loss

**Status:** `not started`

**What it is:**
The current InfoNCE loss treats the positive pair signal as binary — same class or different class. This ignores that classes have semantic relationships: "golden retriever" and "labrador" are more similar than "dog" and "cat", which are more similar than "dog" and "toaster." A hierarchical loss would encode these relationships as a **semantic distance matrix** that defines target distances between class circuit embeddings rather than just pulling same-class pairs together.

**Details:**
Define a semantic distance matrix `S` where `S[i][j]` is the target geometric distance between class `i` and class `j` in circuit space. This can be derived from WordNet/ConceptNet taxonomies, manually specified, or learned from a language model. Replace the binary InfoNCE with a metric learning objective that penalizes deviation from `S`.

Effect in z-space: creates nested circuit neighborhoods (a "dog" trajectory is pulled toward the "canine" cluster, which is part of the "mammal" manifold). Circuits for related classes share lower-level computational structures, directly reducing the total number of unique circuits the model must maintain.

**Progress:**
- [ ] Not started

---

### Circuit Decoder (Generative Inverse-Modeling)

**Status:** `not started`

**What it is:**
Add a decoder `D` trained to reconstruct an input `x` from its circuit embedding `z`. This creates a bidirectional bridge between the model's internal reasoning and human-interpretable visuals. Sampling from cluster centroids in z-space generates images of what the model's circuits consider the "canonical" instance of a class.

**Details:**
Architecture: VAE-style decoder where the latent space is explicitly `z` (the circuit embedding). The decoder is trained end-to-end or post-hoc on a frozen `z` extractor with a reconstruction loss.

Diagnostic capabilities:
- **Centroid sampling:** Sample `z` from the centroid of a class cluster → decode → see the model's internal "prototype" for that class. If the decoded image shows artifacts or spurious features (e.g., a watermark, a specific background color) rather than semantic content, the model's circuit is encoding spurious correlations.
- **Interpolation:** Interpolate between two class centroids in z-space → decode the intermediate points → visualize the model's "ambiguity zone" where its reasoning begins to blend between classes.
- **Anomaly diagnosis:** For a misclassified input, decode its `z` to see what the model's circuits "expected" the input to look like.

**Progress:**
- [ ] Not started

---

### Robustness Auditing via Trajectory Stress-Testing

**Status:** `not started`

**What it is:**
Use the circuit latent space to define a "safety envelope" for each class — the high-probability manifold in z-space where valid reasoning trajectories reside. Track trajectory drift under augmentations or adversarial perturbations. Flag inputs whose trajectories exit the safety envelope even if the final classification is still correct (the "lucky guess vs. correct reasoning" distinction).

**Details:**
- For each class, fit a distribution over z-space using the training set (e.g., a Gaussian or a kernel density estimator).
- At inference: compute the log-likelihood of the input's `z` under the class distribution. Low log-likelihood = trajectory is far from the expected manifold = flag as unreliable.
- Track how log-likelihood changes as augmentation intensity increases (noise σ, rotation angle, brightness shift). Define a "logic breakdown point" as the augmentation level where log-likelihood drops below threshold.
- Applies directly to medical imaging and safety-critical settings where a correct diagnosis reached through a "spurious" circuit pathway is a liability.

**Progress:**
- [ ] Not started

---

### Disambiguation Layer Identification (Predictive Coding)

**Status:** `not started`

**What it is:**
Each class has a mean trajectory `μ_class` — the expected activation pattern at each layer. For a novel input, compare its live trajectory against the expected trajectory and identify the specific layer where the input's trajectory diverges from the class manifold. This "divergence point" tells you exactly where and why the model's reasoning succeeded or failed.

**Details:**
- Compute per-layer distance: `δ_l = ||h_l(x) - μ_class_l||` for the predicted class.
- The divergence point is the layer with the largest jump in `δ_l`.
- Layers before the divergence point reveal which other classes were being "considered" simultaneously (whose manifold overlaps with the input's trajectory at those depths).
- Applications: per-layer audit of model reasoning, identifying at which layer a confusable pair (e.g., "wolf" vs. "dog") actually diverges, debugging misclassifications.

**Progress:**
- [ ] Not started

---

### Targeted Delta-Loss Updates (Surgical Fine-Tuning)

**Status:** `not started`

**What it is:**
Use the divergence point (from Disambiguation Layer Identification above) to identify which specific layers need adjustment when adding a new class or fixing a confusion. Apply update gradients only to those layers, leaving unrelated circuits intact. This is a surgical alternative to full fine-tuning that directly addresses catastrophic forgetting at the circuit level.

**Details:**
- Identify the divergence point layer for the confusion being fixed (e.g., the model confuses "wolf" with "dog" starting at layer 6).
- Apply a delta-loss that specifically penalizes the overlap between the new class and the "intertwined" class at that specific layer depth only.
- Early layers (shared low-level features) are not updated — they don't need to change, and updating them would risk disrupting unrelated circuits.
- Efficiency argument: adding a class that shares circuits with an existing class should only require updating the late layers where differentiation happens, not retraining the full network.

**Dependency:** Requires Disambiguation Layer Identification infrastructure.

**Progress:**
- [ ] Not started

---

### Soft-Weighted Positive Pairs

**Status:** `not started`

**What it is:**
The current positive pair definition (same class label = positive pair) is noisy — the patching validation showed ~50% of same-class pairs don't actually share circuits. Applying a hard attractive force to those pairs distorts the network's natural circuit structure. The fix is to weight the InfoNCE loss by current z-similarity during training rather than filtering by class label alone.

**Details:**
- High z-similarity same-class pairs attract strongly; low z-similarity same-class pairs get near-zero weight
- The multiple valid circuits per class (the "circuit family") then emerge as distinct sub-clusters naturally without being forced to merge
- Start with flat weights (standard InfoNCE behavior) and anneal a sharpening temperature in gradually as z organizes
- This is a dynamic version of the offline patching-derived positive pairs idea — no pre-computation needed, but requires the model's own z-space to have organized enough to be a reliable signal

**Expected outcome:** Higher within-class proxy Spearman ρ against patching ground truth vs. hard class-label version; visible sub-cluster structure in UMAP.

**Dependency:** Benefits from Step 5 infrastructure; conceptually extends the positive pair definition experiments.

**Progress:**
- [ ] Implement soft-weighted InfoNCE (weight each pair by z-similarity, anneal sharpening temperature)
- [ ] Train and compare proxy ρ against hard class-label baseline
- [ ] Check UMAP for within-class sub-cluster emergence

---

### Patching-Derived Positive Pairs (Offline)

**Status:** `not started`

**What it is:**
Rather than using z-similarity to define positive pairs dynamically during training (the soft-weighted approach), an alternative is to pre-compute CircuitSim from activation patching on a baseline model and use those as fixed positive pairs from the start of CTLS training. This gives a cleaner, externally grounded positive pair definition that doesn't depend on the model's current z-space.

**Details:**
- Run patching on a standard ResNet18 trained without CTLS
- Use `CircuitSim > 0.5` as the positive pair criterion; pairs below threshold are excluded from the loss entirely regardless of class label
- Train CTLS with those pairs; compare proxy ρ against class-label and soft-weighted variants
- Limitation: positive pair structure is defined from an unstructured baseline model, which may not be the ideal prior

**Expected outcome:** This experiment directly tests whether the quality of the positive pair definition during training determines the quality of the learned proxy.

**Dependency:** Requires patching harness from Step 2 (complete). More compute than soft-weighted approach.

**Progress:**
- [ ] Run activation patching on unstructured ResNet18 baseline, compute CircuitSim for all pairs
- [ ] Implement CircuitSim-threshold-based pair sampler
- [ ] Train CTLS with offline patching pairs, evaluate proxy ρ
- [ ] Compare against class-label and soft-weighted variants

---

### RoPE-Style Layer Encoding in the Meta-Encoder

**Status:** `not started`

**What it is:**
The transformer meta-encoder (Option B) uses sinusoidal positional encodings to tell the transformer which token came from which layer. This encodes absolute position but doesn't build geometric relationships between layers into the representation itself. RoPE (Rotary Position Embedding) encodes layer position as a rotation in embedding space, meaning the dot product between two layer tokens automatically reflects their relative depth — adjacent layers are geometrically closer than distant layers.

**Details:**
- Implement RoPE positional encoding in the transformer meta-encoder (replacing sinusoidal PE in `transformer_cls`)
- The deeper motivation: the "unraveling" property — if layer position is encoded as rotation angle, you could in principle project a z vector back into per-layer contributions by reversing the rotation, giving decomposed attribution over layers without needing Option A's fixed weighted sum
- This would give Option B's per-input expressivity with Option A's interpretability, resolving the core tradeoff between the two variants
- Evaluate: proxy ρ vs. sinusoidal Option B; test whether rotation-reversal attribution produces per-layer influence scores that correlate with patching ground truth per-layer KL profiles

**Progress:**
- [ ] Implement RoPE layer encoding in transformer_cls MetaEncoder
- [ ] Retrain and compare proxy ρ against sinusoidal Option B
- [ ] Test rotation-reversal per-layer attribution vs. patching KL profiles

---

### Hebbian Co-Activation Regularization

**Status:** `not started`

**What it is:**
InfoNCE is class-discriminative — it pulls same-class circuits together and pushes different-class circuits apart. This creates a risk of distorting circuits that are legitimately shared across classes (curve detectors, texture circuits, which fire for cats, dogs, and birds simultaneously). A Hebbian regularization term would explicitly reward consistent co-activation of layer pairs across the full batch regardless of class label, reinforcing circuit components that are stably used across many inputs.

**Details:**
- Add a Hebbian term that rewards high cross-layer activation correlation across the batch
- This acts as a counterweight to InfoNCE's class-discriminative pressure and structurally protects shared cross-class circuits
- The presence or absence of shared circuit degradation in the current model (measurable from per-layer patching KL profiles on cross-class pairs) determines whether this is urgent to implement
- Experiment: train with and without Hebbian term, check whether shared cross-class circuit integrity is better preserved

**Progress:**
- [ ] Measure per-layer patching KL profiles on cross-class pairs to quantify baseline shared circuit degradation
- [ ] Implement Hebbian co-activation term
- [ ] Compare shared circuit integrity with and without the term

---

### Circuit Economy as a Regularization Objective

**Status:** `not started`

**What it is:**
A network should use the minimum number of distinct computational pathways needed to correctly discriminate its outputs. Redundant circuits — multiple different routings that produce the same prediction — may represent memorization shortcuts that fail under distribution shift. This connects to the grokking literature, where the transition from memorization to generalization corresponds to a transition from redundant to efficient circuits.

**Details:**
- In z-space, circuit economy means penalizing within-class cluster variance — you want same-class inputs to converge toward one canonical circuit (or a small family), not scatter across many
- The soft-weighted positive pairs idea naturally implements a weak version of this
- A stronger version adds an explicit within-class variance penalty on z in each batch
- If circuit economy correlates with generalization, this becomes a principled connection between interpretability and robustness
- Experiment: add variance penalty term, compare proxy ρ and performance under distribution shift against the version without it

**Progress:**
- [ ] Implement within-class z-variance penalty
- [ ] Train with and without penalty, compare proxy ρ
- [ ] Evaluate under distribution shift (noise, augmentation) to test generalization connection
