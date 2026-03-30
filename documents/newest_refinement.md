# Phase 1 Refinement: Flow-Based Circuit Representation

## What Changed and Why

The previous design used **state similarity** as the training signal:

```
s_l(a,b) = sim(h_l(a), h_l(b))
```

This is the cosine similarity of the accumulated representations at layer l. It's the wrong signal for circuit detection because `h_l` is a sum of everything that happened from layer 0 to layer l:

```
h_l = h_0 + delta_1 + delta_2 + ... + delta_l
```

So `sim(h_l(a), h_l(b))` reflects all prior layers blended together. Two inputs could share a circuit at layers 3–5 but look dissimilar at `h_5` because they diverged at layers 1–2. Or they could look similar at `h_5` through completely different circuits. State similarity cannot cleanly isolate where computation was shared.

The correct signal is **flow similarity** — what each block *contributed* to the representation, independent of what accumulated before it. This is the residual delta:

```
delta_l(x) = F_l(x)    # the non-skip branch output of block l, before the addition
```

For a ResNet BasicBlock, the forward pass is:

```python
out = F_l(x)          # main branch (conv -> bn -> relu -> conv -> bn)
out += downsample(x)  # skip connection (identity or projected)
out = relu(out)
```

`F_l(x)` is exactly the transformation the block applied to the input. It is always the same shape as the block output, at every block including dimension-changing transition blocks. This cleanly separates the block's contribution from the accumulated history.

Two inputs share a circuit at a span of layers when they undergo the **same functional transformation** at each block in that span — meaning their `F_l(x)` vectors are similar at each of those layers. This is what the profile should measure.

---

## The New Training Signal

### Flow Vector

For each block l, extract `F_l(x)` via a hook on `bn2` (the last batchnorm in the main branch, before the residual addition). This gives:

```
F_l(x): [B, C_l, H_l, W_l]
```

This is the raw flow at layer l for a batch of inputs.

### Flow Compression

`F_l(x)` is too large to use directly as a reconstruction target — early ResNet18 blocks on CIFAR-10 have shape `[B, 64, 32, 32]` = `[B, 65536]`. Compress it to a tractable fixed-size vector using **adaptive max pooling over a spatial grid**:

```
F_l(x): [B, C_l, H_l, W_l]
    → AdaptiveMaxPool2d(G, G)       # G×G grid, fixed regardless of spatial size
    → [B, C_l, G, G]
    → flatten
    → [B, C_l * G * G]
    → Linear(C_l * G * G, D_flow)  # project to fixed dim
    → flow_l(x): [B, D_flow]
```

**Why max pool, not average pool:** The delta `F_l(x)` is sparse — at most inputs a block makes small changes almost everywhere, with a few spatially localized large activations where it's doing something circuit-relevant. Average pooling washes out those peaks. Max pooling over a grid captures them. The signal you care about is precisely where the delta is large.

**Why a grid, not global max:** Global max pool collapses the entire spatial map to one value per channel, losing all spatial structure. A G×G grid (recommend G=4) retains coarse spatial layout — enough to distinguish "activated top-left" from "activated bottom-right" — at `C_l * 16` dimensions total.

**Why a learned linear projection after:** Different layers have different `C_l` (64, 128, 256, 512 for ResNet18). The linear projection maps each layer's compressed delta to a common `D_flow` dimension. This is a fixed linear layer per layer, not trained — it is part of the target computation pipeline, not the encoder.

**Dimensionality check for ResNet18 with G=4:**
- Blocks 1-2 (layer1): `64 * 4 * 4 = 1024` → project to D_flow
- Blocks 3-4 (layer2): `128 * 4 * 4 = 2048` → project to D_flow  
- Blocks 5-6 (layer3): `256 * 4 * 4 = 4096` → project to D_flow
- Blocks 7-8 (layer4): `512 * 4 * 4 = 8192` → project to D_flow

Recommend `D_flow = 256`. Memory for reconstruction target: `32640 pairs * 256 * 4 bytes ≈ 33 MB` per layer. Tractable.

**Gradient treatment:** The entire flow computation — hook on `bn2`, max pool, flatten, linear projection — runs with `torch.no_grad()`. This is the fixed ground truth signal. No gradients flow through it, same as the backbone activations in the original design.

---

## Hooking Strategy in backbone.py

The current backbone hooks on each BasicBlock's output (post-relu, post-addition). This needs to change to hook on `bn2` inside each block to get `F_l(x)` before the skip addition.

For ResNet18, each BasicBlock has the structure:
```python
# Inside BasicBlock.forward:
out = self.conv1(x)
out = self.bn1(out)
out = self.relu(out)
out = self.conv2(out)
out = self.bn2(out)   # <-- hook here to get F_l(x)
out += residual       # skip connection added after
out = self.relu(out)
```

Register hooks on `block.bn2` for each block across all four layer groups. The hook captures the pre-addition output — the pure block contribution.

**The flow compression pipeline** (AdaptiveMaxPool2d + flatten + linear) runs inside the hook handler or in a separate method that processes the raw `F_l(x)` tensor. Since it runs under `no_grad`, it can live in `FrozenBackbone` as a set of per-layer compression modules that are initialized during the dummy forward pass once layer dims are known.

---

## Training Objective

### Full Objective

```
L = L_recon + lambda * L_geometry
```

Both terms now use the flow signal. The lambda warmup schedule is unchanged.

### L_recon — Flow Reconstruction Fidelity

For each layer l, the regressor takes the element-wise product of two inputs' z-representations and predicts their joint flow compression target:

```
L_recon = (1/L) * sum_l  MSE(regressor_l(z_l(a) ⊙ z_l(b)),  flow_l(a) ⊙ flow_l(b))
```

**Why the element-wise product on the target side:** `flow_l(a) ⊙ flow_l(b)` is the per-dimension co-activation of the two inputs' flow vectors. It is high where both inputs have large delta activations in the same dimension — exactly the signal for shared circuit use. It is symmetric (swapping a and b gives the same vector), matching the symmetry of the prediction input `z_l(a) ⊙ z_l(b)`.

**Regressor output:** Each `regressor_l` maps `[N_pairs, d]` → `[N_pairs, D_flow]`. Since `D_flow` is the same for all layers (after the linear projection in the flow pipeline), the regressor architecture is uniform across layers, but each layer has its own independent regressor instance.

**MSE target shape:** `[N_pairs, D_flow]` — MSE computed element-wise, then averaged over both pairs and dimensions.

### L_geometry — Flow Similarity Organization

The geometry loss organizes z-space so that pairs with similar flows are geometrically close. The scalar similarity target is derived from the flow vectors:

```
s_l(a,b) = sim(flow_l(a), flow_l(b))   # cosine similarity of compressed flow vectors
```

This replaces the previous `sim(h_l(a), h_l(b))` — same geometry loss formula, same soft contrastive structure, but now driven by flow similarity rather than state similarity.

The geometry loss interface is otherwise unchanged — it receives `z_list` and a `[B, B, L]` matrix of scalar flow similarities.

---

## What Changes in the Codebase

### `models/backbone.py` — FrozenBackbone

**Major changes:**

1. Remove the `pool_mode` parameter and all GAP/max/topk pooling logic entirely.

2. Change hook registration from block outputs to `block.bn2` for each block in all four layer groups.

3. Add per-layer flow compression modules as a `nn.ModuleList` inside `FrozenBackbone`. Each module is:
   ```python
   nn.Sequential(
       nn.AdaptiveMaxPool2d((G, G)),   # G=4
       nn.Flatten(),
       nn.Linear(C_l * G * G, D_flow, bias=False)
   )
   ```
   These run under `no_grad` — they are fixed target computation, not trained parameters. Use `requires_grad_(False)` on these modules explicitly.

4. The dummy forward pass now discovers `C_l` per block (from the bn2 output shape) and constructs the compression modules accordingly. `layer_dims` exposed to the MetaEncoder remains `[d, d, ..., d]` — the projection dim — unchanged, since the MetaEncoder interface doesn't change.

5. The trajectory returned from `forward()` remains a list of L L2-normalized tensors of shape `[B, d_projected]`... wait, no. The backbone's `forward()` returns the raw normalized activations to the MetaEncoder. That part is unchanged. What's new is a separate method:
   ```python
   def compute_flow_targets(self, x: Tensor) -> list[Tensor]
   ```
   That runs the full flow pipeline and returns a list of L tensors each `[B, D_flow]`, under `no_grad`. This is called in the trainer to get the reconstruction targets.

   Actually, to avoid running the backbone twice, the hook can capture both `bn2` output (for flow targets) and block output (for trajectory). Or more cleanly: the trainer calls `backbone(x)` once, which populates both `self._trajectory` (block outputs, L2-normalized, for the MetaEncoder) and `self._flow_targets` (bn2 outputs, compressed, for the loss). Both are populated in a single forward pass.

6. `layer_dims` for the MetaEncoder is now derived from the **block output** shapes (same as before, just post-relu), not the bn2 shapes — these are the same for non-transition blocks and the block output shape for transition blocks, which is what the MetaEncoder projectors need.

### `training/unified_trainer.py` — Phase1Trainer

**Changes:**

1. `compute_profiles` is removed or repurposed. The geometry loss scalar targets are now computed from flow vectors:
   ```python
   # For each layer l, compute [B, B] scalar flow similarity matrix
   flow_sim_l = flow_targets_l @ flow_targets_l.T  # after L2-normalizing flow_targets_l
   ```
   Stack into `[B, B, L]` for GeometryLoss.

2. In `_train_epoch`, after calling `backbone(x)` to get the trajectory, also retrieve `backbone._flow_targets` (list of L tensors `[B, D_flow]`).

3. For `InfoLoss`, compute pair flow co-activations:
   ```python
   flow_pairs_a = [f[idx_a] for f in flow_targets]   # list of L x [N_pairs, D_flow]
   flow_pairs_b = [f[idx_b] for f in flow_targets]
   flow_coact   = [flow_pairs_a[l] * flow_pairs_b[l] for l in range(L)]  # list of L x [N_pairs, D_flow]
   ```
   Pass `flow_coact` as the reconstruction target to `InfoLoss`.

4. For `GeometryLoss`, compute scalar similarity matrix from flow targets as described above. Pass `[B, B, L]` to `GeometryLoss` — its interface is unchanged.

5. In `_val_epoch`, geometric consistency (Spearman rho) is now computed between z-space cosine similarity and **flow** cosine similarity, not state cosine similarity.

### `losses/info_loss.py` — InfoLoss

**Changes:**

1. `true_similarities` argument changes from `[N_pairs, L]` scalars to a list of L tensors each `[N_pairs, D_flow]` — the per-layer flow co-activation targets.

2. Each regressor now predicts `[N_pairs, D_flow]` instead of `[N_pairs]`. MSE is computed over the full `D_flow` dimension:
   ```python
   loss_l = mean((regressor_l(z_product) - flow_coact_l) ** 2)
   ```

3. `InfoLoss` now holds a `nn.ModuleList` of per-layer regressors rather than a single shared regressor. Each regressor has its own output head sized to `D_flow`.

### `models/meta_encoder.py` — ProfileRegressor

**Changes:**

1. Add `output_dim` parameter. The final linear layer outputs `[N_pairs, output_dim]`. When `output_dim > 1`, do not squeeze.

2. The trainer constructs one `ProfileRegressor` per layer, each with `output_dim=D_flow`.

### `losses/geometry_loss.py` — GeometryLoss

**No changes.** It receives `z_list` and `[B, B, L]` scalar similarities. The scalars now come from flow similarity instead of state similarity, but the loss formula is identical.

### `evaluation/circuit_analysis.py` — CircuitAnalyzer

**Changes:**

1. `compute_pair_profiles` now computes flow similarity between pairs:
   ```python
   # For each layer l:
   flow_a = flow_targets[l][idx_a]   # [N_pairs, D_flow]
   flow_b = flow_targets[l][idx_b]
   # Scalar similarity for profile:
   profiles[:, l] = F.cosine_similarity(flow_a, flow_b, dim=-1)
   ```

2. Add `compute_flow_targets(images)` method that runs images through backbone and returns the flow targets list.

### `evaluation/metrics.py`

**Changes:**

1. `profile_reconstruction_r2`: update to handle the new target shape `[N_pairs, D_flow]` per layer. Compute R² over the full flattened target, or report mean R² per layer. Update docstring.

### Config YAML files

1. Remove `pool_mode` from all configs — `configs/phase1.yaml` and all ablations.
2. Add `D_flow` and `G` (grid size) as config parameters under `model`:
   ```yaml
   model:
     flow_compression:
       grid_size: 4      # G for AdaptiveMaxPool2d
       flow_dim: 256     # D_flow
   ```
3. The pooling ablation configs (`max_pool.yaml`, `top_k_pool.yaml`) are superseded — note this in their headers and leave them as-is or remove.

### Tests

**`tests/test_losses.py`:**

1. Update `make_true_sims_pairwise` to return a list of L tensors each `[N_pairs, D_flow]` instead of `[N_pairs, L]`.
2. Update `TestInfoLoss` to construct per-layer regressors with the correct `output_dim`.
3. `TestGeometryLoss` is unchanged — it doesn't touch the profile format.

**`tests/test_meta_encoder.py`:**

1. Add `output_dim` to `ProfileRegressor` construction in all tests.
2. Update `test_output_shape` to verify `[N_pairs, output_dim]` output.
3. `test_symmetric_input` still holds — `z_a * z_b == z_b * z_a` regardless of output dim.

---

## Summary of the Core Invariant

The training invariant is now:

```
z_l(a) ⊙ z_l(b)  ≈  flow_l(a) ⊙ flow_l(b)
```

Where `flow_l(x)` is the compressed transformation that block l applied to input x, derived from the non-skip branch output `F_l(x)` via adaptive max pooling and linear projection. The left side lives in `ℝ^d` (the learned circuit space). The right side lives in `ℝ^{D_flow}` (the fixed flow compression space).

The geometry loss ensures that the scalar cosine similarity between `flow_l(a)` and `flow_l(b)` is reflected in the geometric distance between `z_l(a)` and `z_l(b)` in z-space.

Together these two constraints enforce that z-space is both **informative** (encodes what each block did to each input) and **organized** (inputs that share the same block transformation are geometrically close). Circuit discovery by clustering in z-space is then directly recovering groups of inputs that share the same functional transformation sequence across a span of layers.