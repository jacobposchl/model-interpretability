# CTLS Research Roadmap
### Honest State of the Project + What Needs to Happen Next

---

## What This Document Is

This document synthesizes the full set of conceptual issues identified with the current CTLS implementation and lays out the ordered steps needed to turn it into a principled, defensible research project. It is written to be actionable by someone implementing from scratch or restructuring the existing codebase.

The document is organized as: **what we have → what's wrong with it → what to do instead, in order.**

---

## Part 1: The Core Idea (What Is Still Valid)

The fundamental hypothesis of CTLS is correct and worth pursuing:

> A neural network processing semantically similar inputs should use similar internal computational pathways — similar "circuits." If we can measure this, we can use it as a training signal to make models that are interpretable by construction rather than by post-hoc analysis.

The key insight is that a circuit is not a single layer's output — it is the full trajectory of activations across all layers. Two inputs that produce the same final class prediction may have taken completely different internal routes to get there. Capturing that routing structure is the goal.

What we call the "activation trajectory" for input `x` is:

```
T(x) = (h_1(x), h_2(x), ..., h_L(x))
```

where `h_l(x)` is the activation vector at layer `l`. This trajectory is what the project is trying to organize.

---

## Part 2: Terminology Clarification

**Circuit (strict mech interp definition):** The causal subgraph of the weight matrix responsible for a computation — identified via activation patching and causal interventions. Expensive to compute, not differentiable.

**Activation trajectory (what we actually measure):** The ordered list of activation vectors across all layers for a given input. Cheap, differentiable, computable during training.

**The bet the project makes:** If two inputs share a causal circuit, their activation trajectories should be geometrically similar across all layers. Therefore, trajectory similarity is a valid proxy for circuit similarity.

This bet has not been empirically verified. Verifying it is Step 2 of the roadmap below.

---

## Part 3: What Is Wrong With the Current Implementation

### Problem 1: Two disconnected loss signals

The current training objective is:

```
L_total = L_task + λ * L_cons + μ * L_supcon
```

Where:
- `L_cons` = depth-weighted cosine distance between raw per-layer activations `h_l(x1)` and `h_l(x2)` — the meta-encoder is **not involved**
- `L_supcon` = contrastive loss on meta-encoder outputs `z1` and `z2` — depth weighting is **not involved**

These are two completely parallel signals operating on two different things. The consistency loss pushes raw activations to be similar. The SupCon loss pushes the compressed embeddings to cluster by class. Neither informs the other.

The depth weighting lives only in `L_cons` and never touches `z`. The meta-encoder has no structural incentive to care about depth — its input is a flat concatenation of all layers treated equally.

**This means:** the depth weighting rationale and the trajectory compression are architecturally disconnected. You are claiming one unified mechanism but running two separate ones.

### Problem 2: The meta-encoder is purely post-hoc

The meta-encoder produces `z` which is used only in `L_supcon`. But `L_supcon` is just a standard supervised contrastive loss telling `z` to cluster by class label. This makes the meta-encoder functionally identical to any other contrastive representation learner — it is not doing anything specific to circuits or trajectories that a standard SimCLR head operating on the final layer couldn't also do. It is a post-hoc analyzer of the trajectory, not a principled compression of it.

### Problem 3: The evaluation metric is circular

The silhouette score measures how well `z` clusters by class. But `z` was explicitly trained via `L_supcon` to cluster by class. Measuring silhouette on `z` tells you the training objective worked — it tells you nothing about whether `z` is a valid representation of circuits.

This is the core circular evaluation problem: **we defined our metric, optimized for it, and then reported how well we scored on it.**

### Problem 4: The proxy has never been validated

We have never run activation patching experiments to produce ground-truth circuit similarity. Without that ground truth, there is no way to know whether trajectory cosine similarity correlates with actual shared causal subgraphs. The entire project rests on an assumption that has never been tested.

---

## Part 4: The Correct Architecture (Unified Objective)

Replace the two-signal system with a single pipeline where depth weighting is structural, not external.

### Step 1: Per-layer projection to a common dimension

Each layer's activation vector is projected to the same `d`-dimensional space:

```
p_l = Projector_l(h_l)    where p_l ∈ R^d for all l
```

Each `Projector_l` is a small linear layer + LayerNorm + GELU. This handles the fact that different layers have different widths (e.g., ResNet18 goes 64 → 64 → 128 → 128 → 256 → 256 → 512 → 512) and makes the combiner architecture-agnostic.

### Step 2: Depth-aware combination into a single z

Combine `[p_1, ..., p_L]` into a single vector `z` such that later layers contribute more to `z` than earlier layers. Two options in increasing order of expressivity:

**Option A — Fixed weighted sum (simple, interpretable):**
```
z = sum_l(w_l * p_l)    where w_l = l / sum(1..L)
```
Depth weighting is explicit and hand-specified. Cheap. Differentiable. Bakes the inductive bias directly into the representation geometry.

**Option B — Learned attention over layer sequence (expressive, principled):**
Keep `[p_1, ..., p_L]` as a sequence of tokens. Add positional encodings that encode layer depth. Run a small 2-layer transformer. Pool via CLS token to get `z`.

The transformer variant allows the model to learn *which layers matter for which inputs* rather than applying a fixed ramp to everyone. For inputs where class identity becomes clear at layer 5, attention can weight layer 5 more. For harder inputs that need layer 7, it will weight layer 7 more. This is strictly more expressive than a fixed linear ramp and removes an arbitrary design choice.

### Step 3: Single contrastive loss on z

```
L_total = L_task + λ * InfoNCE(z_1, z_2, negatives)
```

Where `z_1` and `z_2` are the depth-aware embeddings of two different images from the same class (positive pair), and negatives are all other-class images in the batch. InfoNCE simultaneously attracts positives and repels negatives, preventing collapse.

**Everything else drops out.** No separate `L_cons`. No separate `L_supcon`. One distance metric on one representation. The depth weighting is a geometric property of how `z` is constructed, not an external penalty.

### Why this is better

- The gradient signal to the backbone is coherent — there is one loss, not two pulling in potentially different directions
- Depth weighting lives inside the representation, not bolted on externally as a loss coefficient
- The meta-encoder is no longer a post-hoc analyzer — it IS the thing being optimized
- The evaluation metric (silhouette on `z`, intraclass ρ) now measures the thing the loss is directly shaping, but validation against ground truth is still required (see Part 5)

---

## Part 5: The Validation Problem — Ordered Roadmap

These steps must be done in order. Each step depends on the previous one.

---

### Step 1: Build the unified objective

Implement the architecture described in Part 4. Specifically:

- Replace the current `MetaEncoder` with a depth-aware version using either fixed weighted sum (Option A) or learned transformer attention (Option B). Start with Option A for simplicity, then try Option B as an ablation.
- Remove `L_cons` from the training objective entirely.
- Remove `L_supcon` from the training objective entirely.
- The new total loss is `L_task + λ * InfoNCE(z_1, z_2)` where `z` comes from the depth-aware meta-encoder.
- Replicate the key metrics from the existing experiments (circuit silhouette, intraclass ρ, noise robustness ratio) on this new objective as a sanity check that it still achieves circuit organization.

**What this gives you:** A cleaner, internally consistent method where the depth weighting and trajectory compression are unified into a single representation. This is the baseline you need before doing any validation experiments.

---

### Step 2: Collect ground-truth circuit similarity via activation patching

This is the foundational validation that the entire project needs. It cannot be skipped.

**What activation patching is:** Take two inputs `x_a` and `x_b`. Run `x_a` through the model. Then run `x_b` but at a specific layer `l`, replace `x_b`'s activations with `x_a`'s activations (patch them in). Measure how much the output changes. If the output barely changes, it means the two inputs were using the same computation at that layer — evidence of a shared circuit. If the output changes a lot, they were using different computations.

**What to build:**
- A patching harness that, given two input images, computes a per-layer causal influence score: how much does patching `x_a`'s layer-`l` activations into `x_b`'s forward pass change `x_b`'s output?
- Aggregate across layers to get a scalar circuit similarity score for each image pair: `CircuitSim(x_a, x_b) ∈ [0, 1]` where 1 means fully shared circuit.
- Run this on a sample of image pairs — e.g., 500 same-class pairs and 500 different-class pairs from the CIFAR-10 test set.

**What this gives you:** A ground-truth pairwise similarity matrix over inputs, grounded in causal intervention rather than geometry. This is the thing `z`-space should be approximating.

**Cost:** This is slow (one patching run per layer per image pair) and non-differentiable, which is why it cannot be a training signal. But it only needs to be run once, offline, on a fixed sample. At CIFAR-10 scale with ResNet18 and 1000 pairs, this is feasible in a few hours on a single GPU.

---

### Step 3: Validate whether trajectory similarity is a valid proxy for circuit similarity

With the ground-truth matrix from Step 2 and the `z` vectors from Step 1, compute:

**Spearman correlation between:**
- Pairwise `z`-space cosine similarity: `cos(z_a, z_b)` for each pair
- Ground-truth causal similarity: `CircuitSim(x_a, x_b)` for each pair

If the correlation is high (e.g., ρ > 0.6), you have empirical evidence that trajectory similarity is a reasonable proxy for actual circuit similarity. The project's central bet is validated.

If the correlation is low, the proxy is not valid and you need to revisit Part 4 — specifically, whether global-average-pooled post-block activations are the right signal, or whether something like attention weights, pre-nonlinearity activations, or gradient-weighted activations would better track causal routing.

**What this gives you:** An honest answer to "is trajectory a valid approximation of circuit?" This is currently the biggest unvalidated assumption in the project.

---

### Step 4: Test whether the current activation extraction is optimal

Once you have the ground-truth patching matrix, you can use it to compare different activation extraction strategies:

- **Current:** Post-block, globally average-pooled activations
- **Alternative A:** Pre-nonlinearity activations (before ReLU/GELU)
- **Alternative B:** Spatially-resolved activations (no average pooling, flatten instead) — captures where in the image the computation is happening
- **Alternative C:** Gradient-weighted activations (similar to GradCAM) — weights activations by how much they influenced the output, which is more directly causal

For each strategy, recompute `z` using the same depth-aware meta-encoder from Step 1, then re-run the Spearman correlation against the patching ground truth.

Whichever strategy produces the highest correlation with actual circuit similarity is the right activation extraction method. This is an empirical question, not one that can be resolved by argument alone.

---

### Step 5: Determine the right clustering structure

Only once you have a validated proxy (Step 3) can you meaningfully ask what the right positive pair definition is. The current implementation uses class labels — two images of the same class are a positive pair. But this bakes in the assumption that circuits cluster by class, which may not be true.

Alternatives to test:

- **Class-level clustering (current):** Same class = positive pair. Works if circuits are class-specific.
- **Feature-level clustering:** Positive pairs are images that share a specific visual feature, regardless of class — e.g., all "fur texture" images, all "circular object" images. Works if circuits are feature-specific and cross class boundaries.
- **Causal similarity clustering (ground truth):** Positive pairs are defined directly from the patching matrix — images with `CircuitSim(x_a, x_b) > threshold` are positives. This is the cleanest definition but requires patching infrastructure from Step 2.

Test each by training the model with that positive pair definition, then evaluating the resulting `z`-space correlation against the patching ground truth. The positive pair definition that produces the highest correlation is the most truthful one.

**What this tells you:** Whether circuits in this model are fundamentally class-organized, feature-organized, or something else. This has direct implications for what CTLS-SSL should use as its positive pair definition when class labels are unavailable.

---

### Step 6: SSL Extension

Only after Steps 1-5 are complete — specifically, after you have a validated proxy and know the right clustering structure — does extending to SSL make sense.

The SSL extension replaces class-label-defined positive pairs with augmentation-proximity-defined positive pairs. Two augmented views of the same image are the minimal version. A stronger version uses learned similarity from the patching ground truth to find pairs that genuinely share circuits rather than just sharing source image identity.

The primary hypothesis for CTLS-SSL (from the existing research synthesis) remains: circuit scaffolding from training data should give the model structural advantages on sample efficiency for semantically related new categories. This can be tested as a downstream evaluation once the core proxy validation is complete.

---

## Part 6: Summary Table

| Step | What It Does | Depends On | Key Output |
|------|-------------|------------|------------|
| 1 | Build unified objective | Nothing | Cleaner method, replicated metrics |
| 2 | Activation patching ground truth | Step 1 | Pairwise circuit similarity matrix |
| 3 | Validate trajectory proxy | Step 2 | Spearman ρ(z-sim, circuit-sim) |
| 4 | Find best activation extraction | Step 3 | Optimal h_l extraction strategy |
| 5 | Find best clustering structure | Step 3 | Class / feature / causal positive pairs |
| 6 | SSL extension | Steps 3–5 | CTLS-SSL with validated foundation |

---

## Part 7: What Remains Valid From the Existing Work

The existing experimental results (silhouette 0.15 → 0.81, intraclass ρ 0.295 → 0.768, noise robustness 0.273 → 0.784) are real and meaningful, but they are results about the internal coherence of the method, not about whether the method is actually measuring circuits. They show:

- The training objective achieves what it was optimized for (class-organized `z` space)
- The resulting representations are more noise-robust than baseline
- Depth-weighting outperforms uniform weighting

These findings survive and become *more* meaningful once the proxy is validated in Step 3 — at that point, you can say not just "we achieved organized trajectory embeddings" but "we achieved organized trajectory embeddings that correspond to actual circuit structure."

The monosemanticity paradox finding (higher circuit silhouette with lower monosemanticity) is also real and interesting regardless — it is empirical evidence that neuron-level and circuit-level organization are measuring genuinely different properties.

---

## Part 8: Open Questions Not Yet Addressed

1. **Does circuit structure by class make sense, or is feature-level the right granularity?** Addressed in Step 5, but the answer is not yet known.
2. **Is global average pooling the right activation extraction?** Addressed in Step 4.
3. **Is a fixed linear depth ramp better or worse than learned attention weighting?** Ablation within Step 1.
4. **Does CTLS-SSL's sample efficiency hypothesis hold?** Step 6 — requires Steps 1-5 first.
5. **Does the monosemanticity paradox (structured superposition > monosemantic features for circuit organization) generalize beyond CIFAR-10 / ResNet18?** Future work, out of scope for current phase.