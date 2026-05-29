# Halo-VLA World Model — Technical Reference

This document covers the full design of the DiT-based visual world model in Halo-VLA:
per-frame conditioning, Classifier-Free Guidance, the Heun ODE solver, auxiliary training
losses, gradient accumulation, and the bug fixes applied to make it all work correctly.

---

## Table of Contents

1. [Overview](#overview)
2. [Per-Frame Conditioning Pipeline](#per-frame-conditioning-pipeline)
   - [World Action Tokens](#world-action-tokens)
   - [Frame Index Positional Embedding](#frame-index-positional-embedding)
   - [Context Frame Pixel Encoder](#context-frame-pixel-encoder)
   - [Gated DiT Conditioning](#gated-dit-conditioning)
   - [Classifier-Free Guidance Dropout](#classifier-free-guidance-dropout)
3. [Inference: Heun's 2nd-Order ODE Solver](#inference-heunsm-2nd-order-ode-solver)
4. [Training Losses](#training-losses)
   - [Conditional Flow Matching (CFM)](#conditional-flow-matching-cfm)
   - [VGG Perceptual Loss](#vgg-perceptual-loss)
   - [SSIM Loss](#ssim-loss)
   - [Temporal Smoothness Loss](#temporal-smoothness-loss)
5. [Training: Gradient Accumulation](#training-gradient-accumulation)
6. [Checkpoint Resume](#checkpoint-resume)
7. [Bug Fixes](#bug-fixes)
8. [Configuration Reference](#configuration-reference)

---

## Overview

The world model predicts a sequence of `num_predict_frames` future RGB frames from:

- `N` observed context frames (processed by ViT + decoder transformer)
- The hidden state at each `<halo_world_video>` token position — one token per predicted frame
- An optional last observed frame for pixel-level scene anchoring

Each frame is generated independently by a shared Diffusion Transformer (DiT) conditioned on a
**per-frame embedding** that bundles semantic, positional, and pixel-level scene signals.

**Key classes:**

| Class / function | File | Role |
|---|---|---|
| `RGBFramePredictor` | `models/dit_frame_prediction.py` | Top-level predictor; owns cross-attention, world action tokens, DiT |
| `DiT` | `models/DiT.py` | Core transformer with adaLN-Zero modulation blocks |
| `_create_all_frame_contexts` | `dit_frame_prediction.py` | Builds per-frame conditioning vectors (batched) |
| `compute_loss` | `dit_frame_prediction.py` | CFM + perceptual + SSIM + temporal loss |
| `predict_future_frames` | `dit_frame_prediction.py` | Heun ODE + CFG inference |

---

## Per-Frame Conditioning Pipeline

Each predicted frame `i` receives a unique conditioning vector built from four stacked signals.
All four are computed in `_create_all_frame_contexts` in a single batched cross-attention call
(no Python loop over frames).

```
world_action_tokens[frame_offset + i]       ← learnable, one per global frame index
        │
        ▼
  MultiheadAttention  (Q = action token, K/V = text embeddings)
        │  residual + LayerNorm + FFN + LayerNorm
        ▼
  base_cond  [BF, D]
        │
        + frame_pos_proj( sinusoidal(frame_offset + i) )    ← unambiguous frame identity
        │
        + context_frame_enc( last_observed_frame )          ← SVD-style pixel anchor
        │
    CFG dropout: replace with cfg_null_context  (p=0.1, training only)
        │
        ▼
  DiT adaLN: c = sigmoid(W_gate · ctx) · t_emb + ctx_emb   ← gated fusion
```

### World Action Tokens

```python
self.world_action_tokens = nn.Parameter(
    torch.empty(num_predict_frames, context_dim)
)
nn.init.normal_(self.world_action_tokens, std=0.02 * (num_predict_frames ** 0.5))
```

One learnable token per predicted frame. The higher init std (`0.02 * sqrt(F)`) ensures tokens
start distinguishable from each other — critical for per-frame diversity before training gives the
model another source of separation.

At training and inference, `frame_offset` selects the correct token range:

```python
token_start = frame_offset
token_end   = min(frame_offset + num_frames, self.num_predict_frames)
action_query = self.world_action_tokens[token_start:token_end]
```

This fixed a silent bug where the autoregressive loop always picked `world_action_tokens[0]`
regardless of which frame was being generated (because `num_frames=1` was passed per step).

### Frame Index Positional Embedding

World action tokens can be similar early in training. The sinusoidal frame index embedding
gives the DiT an **explicit, unambiguous** signal about which frame it is generating, independent
of the token's learned value.

```python
def _frame_pos_embedding(frame_indices: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=frame_indices.device)
        * (math.log(10000.0) / max(half - 1, 1))          # ← math.log, not torch
    )
    args = frame_indices.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

`math.log` is used instead of `torch.log(torch.tensor(...))` to avoid creating a CPU tensor
that causes a device mismatch when `frame_indices` lives on GPU (see [Bug Fixes](#bug-fixes)).

The embedding is projected through a learned linear layer (`frame_pos_proj`) and added to
`base_cond` so the model can learn to weight it relative to the semantic conditioning.

### Context Frame Pixel Encoder

Production video diffusion models (SVD, VideoLDM) condition the denoiser on the last observed
frame at the pixel level. Without this, the DiT has no direct scene anchor — it must reconstruct
the current state entirely from a pooled transformer embedding, which is a much harder task.

```python
self.context_frame_enc = nn.Sequential(
    nn.Conv2d(rgb_ch, context_dim // 4, kernel_size=8, stride=8),   # 224 → 28
    nn.GELU(),
    nn.Conv2d(context_dim // 4, context_dim // 2, kernel_size=4, stride=4),  # 28 → 7
    nn.GELU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(context_dim // 2, context_dim),
    nn.LayerNorm(context_dim),
)
```

The encoder is deliberately lightweight — two strided convs into a global average pool —
so it does not dominate the conditioning vector. Its output is added directly to `base_cond`.

During multi-frame generation the context frame for frame `i` is the previously predicted (or
last observed) frame, building an autoregressive pixel chain:

```python
prev_frames = torch.cat(
    [last_context_frame.unsqueeze(1), future_rgb[:, :-1]], dim=1
)  # [B, F, 3, H, W]
```

### Gated DiT Conditioning

The original conditioning fused timestep and context additively: `c = t_emb + ctx_emb`.
Under mixed-precision training the large timestep embedding magnitude tends to swamp the smaller
context signal, making the DiT effectively unconditional early in training.

The fix is a learned sigmoid gate that lets the context multiplicatively modulate the timestep
signal, and then adds the context directly:

```python
# In DiT.forward:
gate = torch.sigmoid(self.context_gate(context))   # [B, D] ∈ (0, 1)
ctx  = self.context_embed(context)                 # [B, D]
c    = gate * t_emb + ctx
```

`context_gate.bias` is initialised to zero so `sigmoid(0) = 0.5` at the start —
equal weighting of timestep and context — and the gate is free to specialise during training.

### Classifier-Free Guidance Dropout

During training, a fraction of samples have their conditioning replaced with a learned null
vector. At inference, the model is run twice (conditioned and unconditioned) and the outputs
are extrapolated:

```python
v_cfg = v_null + guidance_scale * (v_cond - v_null)
```

**Training:**
```python
cfg_drop = torch.rand(B, device=device) < cfg_dropout_prob   # [B] bool
# Inside _create_all_frame_contexts:
drop_flat = cfg_drop.unsqueeze(1).expand(-1, num_frames).reshape(BF)
null = self.cfg_null_context.unsqueeze(0).expand(BF, -1)
cond = torch.where(drop_flat.unsqueeze(1), null, cond)
```

`cfg_null_context` is a learned `nn.Parameter` (not fixed zeros) so the model can find the
most informative null embedding for its specific distribution.

**Inference (Heun step):**
```python
if use_cfg:
    v1_null = self.dit_rgb(x, t, cond_null)
    v1 = v1_null + guidance_scale * (v1 - v1_null)
```

---

## Inference: Heun's 2nd-Order ODE Solver

The Euler solver used previously accumulates `O(dt)` error per step, leading to visible
drift at low step counts. Heun's method (the explicit trapezoidal rule) uses a predictor-corrector
that reduces the local truncation error to `O(dt²)`:

```
x_pred  = x  + v(x, t)  · dt          # Euler predictor
v2      = v(x_pred, t + dt)            # corrector velocity at the predicted point
x_next  = x  + (v(x, t) + v2) / 2 · dt   # trapezoidal average
```

**Implementation in `predict_future_frames`:**

```python
for i in range(num_steps):
    t_val = i * dt
    t     = torch.full((BF,), t_val, device=x.device)

    v1 = self.dit_rgb(x, t, cond)
    if use_cfg:
        v1_null = self.dit_rgb(x, t, cond_null)
        v1 = v1_null + guidance_scale * (v1 - v1_null)

    x_next = x + v1 * dt
    t_next = torch.full((BF,), min(t_val + dt, 1.0 - 1e-5), device=x.device)  # ← clipped

    v2 = self.dit_rgb(x_next, t_next, cond)
    if use_cfg:
        v2_null = self.dit_rgb(x_next, t_next, cond_null)
        v2 = v2_null + guidance_scale * (v2 - v2_null)

    x = x + (v1 + v2) * 0.5 * dt
```

`t_next` is clipped to `1.0 - 1e-5` on the last step to avoid querying the model at `t=1`,
which is outside the training distribution (`t ~ Uniform(0, 1)`).

With CFG enabled, each Heun step requires **4 DiT forward passes** (v1_cond, v1_null, v2_cond,
v2_null). For fast inference, disable CFG (`guidance_scale=1.0`) to halve the compute cost.

---

## Training Losses

### Conditional Flow Matching (CFM)

The primary loss. The model learns the velocity field `v*(x_t, t) = x₁ - x₀` along the optimal
transport path from Gaussian noise to the target frame:

```
x₀ ~ N(0, I)
t  ~ U(0, 1)
x_t = (1 - t) · x₀ + t · x₁
v*  = x₁ - x₀

L_cfm = MSE( DiT(x_t, t, cond), v* )
```

### VGG Perceptual Loss

Pixel-space MSE penalises all deviations equally, which produces blurry predictions in
expectation. The perceptual loss instead matches VGG feature activations (relu1_2, relu2_2),
which are sensitive to edges and textures rather than raw pixel values.

The loss is computed on the **estimated clean frame**, recovered from the noisy sample without
running the full ODE:

```
x₁_hat = x_t + (1 - t) · v_pred    (exact when v_pred == v*)
L_perc  = MSE(VGG(x₁_hat), VGG(x₁))
```

The VGG model runs in **float32** even under autocast. Both `x₁_hat` and `x₁` are cast to
`.float()` before the VGG forward to avoid a dtype mismatch (float16 inputs vs float32 VGG
weights):

```python
perc_loss = vgg(x1_hat.float(), x1_tgt.float())
```

### SSIM Loss

Structural Similarity penalises blurring more aggressively than MSE by measuring luminance,
contrast, and structure:

```
L_ssim = 1 - SSIM(x₁_hat, x₁)
```

The SSIM kernel is an 11×11 Gaussian computed per forward call. The `kernel.expand()` must
be followed by `.contiguous()` because PyTorch's grouped `F.conv2d` requires a contiguous
weight tensor (a non-contiguous view raises `RuntimeError`):

```python
kernel = kernel.expand(C, 1, window_size, window_size).contiguous()
```

### Temporal Smoothness Loss

When predicting a sequence of frames, independent per-frame optimisation can produce
flickering motion — large velocity swings from frame to frame. The temporal loss penalises
this by requiring consecutive velocity fields to be similar:

```python
def F_func_temporal(v_frames: torch.Tensor) -> torch.Tensor:
    """v_frames: [B, F, C, H, W]"""
    return F.mse_loss(v_frames[:, 1:], v_frames[:, :-1])
```

**Combined loss:**

```python
total = (
    cfm_loss
    + perceptual_weight * perc_loss   # default 0.1
    + ssim_weight       * ssim_loss   # default 0.4
    + temporal_weight   * temp_loss   # default 0.2
)
```

The weights can be set per-run via CLI flags. Setting a weight to `0` disables that component
entirely with no performance overhead (the loss tensor is never computed).

---

## Training: Gradient Accumulation

When GPU memory is the bottleneck, gradient accumulation simulates a larger effective batch by
accumulating gradients across `N` micro-batches before each optimizer step.

**Effective batch size:**
```
effective_batch = batch_size × grad_accum_steps
```

**How it works in the training loop:**

```python
optimizer.zero_grad()   # reset once before the accumulation window

for step, batch in enumerate(train_loader, 1):
    with autocast("cuda"):
        total_loss = ...

    # Divide by accum_steps so accumulated gradients equal the *average*
    # over the effective batch, not the sum.
    scaler.scale(total_loss / accum_steps).backward()

    is_last_batch = (step == len(train_loader))
    if step % accum_steps == 0 or is_last_batch:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
```

Key points:
- `optimizer.zero_grad()` is called **after** each optimizer step, not before each micro-step.
- `global_step` increments only on optimizer steps, so logging and visualisation cadences
  remain consistent regardless of accumulation depth.
- Gradient clipping runs on the **fully accumulated gradient** (after `scaler.unscale_`),
  ensuring the clip threshold is relative to the effective batch, not a micro-batch.

**Learning rate scaling:** when increasing the effective batch, scale the LR linearly
(or by `sqrt`) to preserve training dynamics:

```
# doubling effective batch → double LR
--grad_accum_steps 16 --lr 4e-4   # vs baseline --grad_accum_steps 1 --lr 1e-4
```

**Cosine scheduler:** the scheduler's `T_max` should be in optimizer steps, not micro-steps.
If you change `accum_steps` significantly, adjust `T_max`:

```python
# Current (train.py)
T_max = epochs * len(train_loader)   # in micro-steps

# Correct for accumulation
T_max = epochs * len(train_loader) // accum_steps   # in optimizer steps
```

---

## Checkpoint Resume

Training can be resumed from any saved checkpoint:

```bash
python scripts/train.py --resume checkpoints/run1/halo_vla_epoch10.pt ...
```

The checkpoint stores: model weights, optimizer state, scheduler state, AMP scaler state,
`epoch`, and `global_step`. On resume all five are restored and training continues from
`start_epoch = checkpoint["epoch"] + 1`.

**PyTorch 2.6 compatibility:** `torch.load` with `weights_only=True` (the new default)
rejects custom objects unless they are explicitly allow-listed. `HaloVLMConfig` is a Python
dataclass stored inside the checkpoint and must be declared safe:

```python
with torch.serialization.safe_globals([HaloVLMConfig]):
    ckpt = torch.load(path, map_location=device, weights_only=True)
```

---

## Bug Fixes

The following bugs were identified by code review and fixed.

### 1 — CPU tensor device mismatch in `_frame_pos_embedding`

**Location:** `dit_frame_prediction.py`, `_frame_pos_embedding`

**Problem:** `torch.log(torch.tensor(10000.0))` creates a CPU scalar tensor. When multiplied
with `frame_indices` on GPU, PyTorch raises a device mismatch error.

```python
# Before (broken on GPU)
* (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))

# After
* (math.log(10000.0) / max(half - 1, 1))
```

`math.log` returns a Python float and participates in the tensor computation without
materialising a CPU tensor.

### 2 — Non-contiguous SSIM kernel in grouped `F.conv2d`

**Location:** `dit_frame_prediction.py`, `_ssim_loss`

**Problem:** `kernel.expand(C, 1, W, W)` returns a non-contiguous view. PyTorch's grouped
convolution (`groups=C`) requires a contiguous weight tensor; passing the view raises:
```
RuntimeError: expected contiguous weight tensor
```

```python
# Before
kernel = kernel.expand(C, 1, window_size, window_size)

# After
kernel = kernel.expand(C, 1, window_size, window_size).contiguous()
```

### 3 — Dtype mismatch between VGG weights and AMP inputs

**Location:** `dit_frame_prediction.py`, `compute_loss`

**Problem:** Inside `autocast("cuda")`, intermediate tensors are float16. The frozen VGG model
has float32 weights. Feeding float16 activations into a float32 linear/conv raises a dtype error.

```python
# Before
perc_loss = vgg(x1_hat, x1_tgt)

# After
perc_loss = vgg(x1_hat.float(), x1_tgt.float())
```

### 4 — `F` variable shadowed `torch.nn.functional` in `compute_loss`

**Location:** `dit_frame_prediction.py`, `compute_loss`

**Problem:** Unpacking `B, F, C, H, W = future_rgb.shape` at function scope shadows the
module-level `import torch.nn.functional as F`. Any call to `F.mse_loss`, `F.conv2d`, etc.
inside `compute_loss` after that line would silently call an integer instead, raising
`TypeError: 'int' object is not callable` on the first use.

```python
# Before
B, F, C, H, W = future_rgb.shape

# After
B, nF, C, H, W = future_rgb.shape
```

### 5 — Heun solver queries model at `t = 1.0`

**Location:** `dit_frame_prediction.py`, `predict_future_frames`

**Problem:** On the last Heun step, `t_next = t_val + dt = 1.0`. The model was trained with
`t ~ Uniform(0, 1)` — querying at `t=1` is outside the training distribution, producing
garbage velocity estimates that corrupt the final integration step.

```python
# Before
t_next = torch.full((BF,), t_val + dt, device=x.device)

# After
t_next = torch.full((BF,), min(t_val + dt, 1.0 - 1e-5), device=x.device)
```

### 6 — Mixed batch `future_frames` in `eo_collate_fn`

**Location:** `dataloader/eo_dataset.py`, `eo_collate_fn`

**Problem:** The EO dataset only returns `future_frames` for samples that have enough images
(`n_total > n_predict`). The collate function checked only `batch[0]`; if the first sample
lacked `future_frames` but later samples had it, those frames were silently dropped. Conversely,
if `batch[0]` had it but a later sample did not, stacking raised `KeyError`.

```python
# Before — only checks batch[0]
if "future_frames" in batch[0]:
    out["future_frames"] = torch.stack([b["future_frames"] for b in batch], dim=0)

# After — only stacks when ALL items have the key (guarantees uniform shapes)
items_with_future = [b["future_frames"] for b in batch if "future_frames" in b]
if len(items_with_future) == B:
    out["future_frames"] = torch.stack(items_with_future, dim=0)
```

---

## Configuration Reference

All world-model hyperparameters flow through `HaloVLMConfig` (defined in `config/model_config.py`).
Training-time weights for auxiliary losses are passed as CLI arguments and forwarded to
`compute_loss` without touching the config.

| Config field | Default | Description |
|---|---|---|
| `num_visual_predict_frames` | 5 | Max frames `RGBFramePredictor` is built for |
| `dit_hidden_size` | 512 | DiT transformer hidden dimension |
| `dit_depth` | 8 | Number of DiT transformer blocks |
| `dit_num_heads` | 8 | Attention heads in each DiT block |
| `dit_patch_size` | 4 | Spatial patch size (pixels) |
| `dit_max_resolution` | 256 | Maximum spatial resolution the pos-embed supports |
| `dit_mlp_ratio` | 4.0 | MLP hidden / hidden ratio in DiT blocks |
| `dit_num_sample_steps` | 50 | ODE integration steps at inference |
| `dit_time_freq_dim` | 256 | Frequency embedding dimension for timestep |
| `emb_dim` | 512 | Shared embedding dim (DiT context, cross-attn, etc.) |

| CLI flag | Default | Description |
|---|---|---|
| `--num_predict_frames` | 5 | Frames to predict per sample |
| `--perceptual_weight` | 0.1 | VGG perceptual loss coefficient |
| `--ssim_weight` | 0.4 | SSIM loss coefficient |
| `--temporal_weight` | 0.2 | Temporal smoothness loss coefficient |
| `--grad_accum_steps` | 1 | Gradient accumulation depth |
| `--resume` | — | Checkpoint path to resume from |
