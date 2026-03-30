Here's a Claude Code prompt you can paste directly:

---

## Context

This is a mechanistic interpretability project. A frozen backbone (ResNet18) processes inputs and produces activation trajectories `T(x) = (h_1, ..., h_L)`. A meta-encoder — a RoPE transformer — reads those trajectories and produces per-layer encoded representations `z_1, ..., z_L`. The goal is for the geometry of z-space to reflect pairwise computational similarity across layers, such that circuits (recurring computational pathways spanning contiguous layer ranges) can be discovered by clustering in that space.

The training signal is derived entirely from the backbone's own activation structure — no class labels. For any two inputs `a` and `b`, the **alignment profile** is the ground truth signal describing how similar their activations are at each layer.

## The Problem Being Fixed

The current alignment profile collapses per-channel activation similarity into a single scalar per layer:

```
s_l(a,b) = dot(normalize(h_l_a), normalize(h_l_b))  →  scalar ∈ ℝ
P(a,b) = [s_1, s_2, ..., s_L]  →  ℝ^L
```

This causes two problems:

1. Two input pairs can have identical scalar similarity at a layer but completely different per-channel agreement patterns. The scalar discards all information about *which* channels the two inputs agree on — only *how much* they agree overall.
2. On pretrained ResNet18 with CIFAR-10, the population of scalar similarities is compressed into a narrow high band (≈0.85–0.95 across all layers), making Criterion 3 (within-span elevation) structurally unsatisfiable — clusters can't be elevated above a population that already has no spread.

## The New Formulation: Rich Per-Channel Profile

Replace the scalar profile with a per-channel co-activation vector at each layer:

```
s_l_rich(a,b) = normalize(h_l_a) ⊙ normalize(h_l_b)  →  ℝ^{D_l}
```

Where `⊙` is element-wise multiplication of the two L2-normalized activation vectors. Note: `sum(s_l_rich(a,b))` recovers the old scalar, so this is a strict generalization — no information is lost, only gained.

This tells you *which channels* the two inputs co-activate at layer l, not just whether they co-activate overall. Two pairs with identical scalar similarity but different per-channel patterns are now correctly distinguished.

**Important:** Do NOT apply GAP or any spatial pooling before computing the rich profile. The activations `h_l` should be flattened from `[B, C, H, W]` to `[B, C*H*W]` and then L2-normalized. This preserves spatial co-activation structure. The rich profile at layer l then has dimensionality `D_l = C_l * H_l * W_l`. For ResNet18 on CIFAR-10, layer dims will be larger than before (e.g. the final block: 512*4*4 = 8192) but this is acceptable since the rich profile is only used as a training target, not stored in full for all pairs.

## What Changes in the Codebase

### 1. `models/backbone.py` — `FrozenBackbone`

- Remove all pooling logic (`_pool_spatial`, `pool_mode` parameter, GAP/max/topk options).
- In `_make_hook`: instead of pooling spatial dims, **flatten** the tensor from `[B, C, H, W]` to `[B, C*H*W]`, then L2-normalize. For ViT CLS tokens (already `[B, D]`), behavior is unchanged.
- `layer_dims` discovered by the dummy forward pass will now reflect the full flattened spatial dimensions.
- The `pool_mode` config key in all YAML files will be ignored/removed — update accordingly.

### 2. `training/unified_trainer.py` — `Phase1Trainer.compute_profiles`

The current method returns `[B, B, L]` of scalars. Replace with a method that returns the rich profile.

The rich profile for a batch is: for each layer `l`, compute `normalize(h_l)[i] ⊙ normalize(h_l)[j]` for all pairs `(i, j)`. 

For training, you don't need to store `[B, B, L, D_l]` — that's too large. Instead, compute the rich profile **on-the-fly for the specific pairs used in the loss**, not the full `[B, B]` matrix. The trainer already extracts upper-triangle pair indices `idx_a, idx_b`. So compute:

```python
# For each layer l:
rich_sim_l(a,b) = h_l[idx_a] * h_l[idx_b]  # element-wise, shape [N_pairs, D_l]
# Stack across layers into a list of length L, each [N_pairs, D_l]
```

This list of per-layer rich similarity vectors is the new ground truth for `InfoLoss`.

For `GeometryLoss`, derive a scalar summary from the rich profile by taking the **mean** of the per-channel co-activation vector: `mean(h_l[i] * h_l[j], dim=-1)` → scalar. This is exactly the old scalar cosine similarity, but now computed from the already-flattened (not GAP'd) activations, which have better spatial resolution. Pass this `[B, B, L]` scalar matrix to `GeometryLoss` as before — its interface doesn't change.

### 3. `losses/info_loss.py` — `InfoLoss`

Current: predicts a scalar per layer, targets `true_similarities[:, l]` of shape `[N_pairs]`.

New: predicts a vector of shape `[N_pairs, D_l]` per layer, targets the rich per-channel similarity `[N_pairs, D_l]`.

The `ProfileRegressor` takes `z_l_a ⊙ z_l_b` of shape `[N_pairs, d]` as input. Its output head currently produces `[N_pairs, 1]` (squeezed to `[N_pairs]`). It needs to now produce `[N_pairs, D_l]`. Since `D_l` varies per layer, the regressor cannot have a single fixed output head for all layers.

**Options:** 
- Make `ProfileRegressor` take `output_dim` as a parameter and construct layer-specific regressors — one per layer, each with its own output projection head. This is the cleanest design.
- Store them as a `nn.ModuleList` of `ProfileRegressor` instances, one per layer, in the trainer and `InfoLoss`.

The MSE loss becomes:
```python
loss_l = mean((predicted_l - rich_sim_l) ** 2)  # both [N_pairs, D_l]
```

### 4. `models/meta_encoder.py` — `ProfileRegressor`

Add an `output_dim` parameter (default 1 for backward compat or remove default). The final linear layer should output `[N_pairs, output_dim]`. When `output_dim > 1`, do not squeeze. Update accordingly.

### 5. `evaluation/circuit_analysis.py` — `CircuitAnalyzer`

`compute_pair_profiles` currently returns `[N_pairs, L]` scalars. Add a new method `compute_pair_rich_profiles` that returns a list of length `L`, each element `[N_pairs, D_l]` — the per-channel co-activation vectors. The old scalar method can be retained for the geometry loss scalar path and for metrics/evaluation.

### 6. `evaluation/metrics.py` — Criterion 1

`profile_reconstruction_r2` currently compares `[N_pairs, L]` predicted vs true scalars. With the rich profile, reconstruction accuracy should be computed over the full `[N_pairs, sum(D_l)]` concatenated vectors, or reported as mean R² per layer. Update the function signature and docstring to handle both cases, or add a separate `rich_profile_reconstruction_r2` function.

### 7. Config YAML files

Remove `pool_mode` from all configs (`configs/phase1.yaml`, `configs/ablations/max_pool.yaml`, `configs/ablations/top_k_pool.yaml`, `configs/ablations/info_only.yaml`, `configs/ablations/geometry_only.yaml`). The pooling ablation experiment (`max_pool.yaml`, `top_k_pool.yaml`) is now superseded by this change — note this in comments.

## What Does NOT Change

- The meta-encoder architecture (`MetaEncoder`, `RoPETransformerLayer`, `RotaryPositionEmbedding`) is unchanged. It still takes a trajectory list and outputs `z_1, ..., z_L`.
- `GeometryLoss` is unchanged — it still takes `z_list` and a `[B, B, L]` scalar similarity matrix.
- The span-centric discovery pipeline (`SpanCentricDiscovery`) is unchanged.
- The training loop structure in `Phase1Trainer` is unchanged — only the profile computation and loss targets change.
- Criteria 2–5 are unchanged.
- All tests in `test_meta_encoder.py` and `test_discovery.py` are unchanged.

## Tests to Update

`tests/test_losses.py`:
- `TestInfoLoss`: `make_true_sims_pairwise` currently returns `[N_pairs, L]` scalars. Update to return a list of length `L`, each `[N_pairs, D_l]` to match the new `InfoLoss` interface. Update all `InfoLoss` tests accordingly.
- The `ProfileRegressor` in test helpers needs an `output_dim` per layer — pick a small fixed dim (e.g. 16) for testing.

## Summary of the Invariant Being Preserved

The core training invariant is:

```
enc_l(a) ⊙ enc_l(b)  ≈  normalize(act_l(a)) ⊙ normalize(act_l(b))
```

Where `act_l` is the **flattened, un-pooled** activation at layer `l`. The left side lives in the learned low-dimensional circuit space `ℝ^d`. The right side lives in the high-dimensional but now spatially-aware activation space `ℝ^{D_l}`. The InfoLoss regressor learns the projection from `ℝ^d` to `ℝ^{D_l}` that makes these match. The geometry loss ensures the scalar summary of the rich profile (mean of the right side) organizes the z-space correctly for clustering.