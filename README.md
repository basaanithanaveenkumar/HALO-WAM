<div align="center">

# Halo-VLA

### A compact Vision-Language-Action model with an integrated world model

*Perceive · Reason · Act · Imagine*

</div>

---

Halo-VLA is a single unified model that handles **language grounding**, **robot action prediction**, and **future-frame imagination** in one forward pass. It takes camera frames and proprioceptive state as input, generates a natural-language description of the scene, predicts a continuous action trajectory, and synthesises what the world will look like next — all jointly trained end-to-end.

<div align="center">

| Training visualisation | Diffusion (noise → frame) |
|:---:|:---:|
| ![Training visualisation](assets/training_viz.gif) | ![DiT diffusion](assets/diffusion_viz.gif) |
| Context scroll → text reveal → action trajectory → future frames | DiT velocity-field denoising: pure noise → predicted RGB frame |

</div>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Halo-VLA                                  │
│                                                                     │
│  Images [B,N,3,H,W] ──► ViT encoder ──► patch embeddings           │
│  State  [B,S,D_s]   ──► State MLP   ──► state embeddings           │
│  Text tokens        ──► Factored embedding table                    │
│           │                                                         │
│           └──────► Decoder Transformer (MoE) ◄────────────────┐    │
│                         8 layers · 16 heads                    │    │
│                         DeepSeek MoE FFN                       │    │
│                    ┌────────┬────────┬──────────┐              │    │
│                    │        │        │          │              │    │
│                 LM head  Action   Visual   World video         │    │
│                 (text)   hiddens  context   query hiddens      │    │
│                    │        │        │          │              │    │
│                 Language  Flow    DiT RGB   (conditioning)     │    │
│                  loss    Matching  head                        │    │
│                          Decoder  (future                      │    │
│                          (actions) frames)                     │    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key components

| Module | What it does |
|---|---|
| **ViT** | 4-layer Vision Transformer encodes each frame into patch embeddings |
| **Decoder Transformer** | 8-layer causal transformer with DeepSeekMoE FFN; fuses vision, state and text |
| **Flow Matching Action Decoder** | Predicts robot trajectories via ODE integration; learns the velocity field `v*(x,t) = x₁ − x₀` |
| **DiT World Model** | Diffusion Transformer predicts future RGB frames conditioned on world-action tokens and text embeddings |
| **ALBERT-style factored embeddings** | Vocabulary embedding bottlenecked through a 128-dim projection; cuts embedding memory ~10× vs full tables |

### Flow matching — how action prediction works

```
Training                              Inference
──────────────────────────────────    ──────────────────────────
x₀ ~ N(0, I)  (noise)                x  = x₀  (start from noise)
x₁ = target action chunk             dt = 1 / num_steps
t  ~ U(0, 1)                         for i in range(num_steps):
x_t = (1-t)·x₀ + t·x₁                   v = DiT(x, t, cond)
v*  = x₁ − x₀                            x = x + v · dt
loss = MSE(DiT(x_t, t, cond), v*)    return x  # ≈ clean action chunk
```

The same formulation drives the **DiT world model** — it learns to denoise random noise into a predicted future video frame, conditioned on the decoded world-action tokens.

---

## Project structure

```
Han-WAM/
├── src/Halo_VLA/models/
│   ├── halo_vla.py              # Main model: forward, action loss, visual loss
│   ├── vit.py                   # Vision Transformer encoder
│   ├── transformer.py           # Decoder transformer blocks
│   ├── moe.py                   # DeepSeek Mixture-of-Experts FFN
│   ├── flow_action_decoder.py   # Flow matching action decoder
│   ├── dit_frame_prediction.py  # DiT world model (future frame prediction)
│   ├── DiT.py                   # Core Diffusion Transformer with adaLN
│   ├── lm_head.py               # Language model head
│   └── state_encoder.py         # Proprioceptive state encoder
├── config/
│   ├── model_config.py          # HaloVLMConfig dataclass (all hyperparameters)
│   ├── tokens.py                # Special token definitions
│   └── special_tokens.json      # Token IDs for <image>, <action>, <state>, <world_video>
├── dataloader/
│   ├── airoa_moma_dataset.py    # AiroaMoMa video dataset (lerobot parquet format)
│   └── eo_dataset.py            # EO dataset loader
├── scripts/
│   ├── train.py                 # Training loop with visualisation
│   └── visualize.py             # Standalone inference visualiser
└── assets/
    ├── training_viz.gif         # Training visualisation example
    └── diffusion_viz.gif        # DiT denoising example
```

---

## Installation

```bash
git clone https://github.com/basaanithanaveenkumar/Han-WAM.git
cd Han-WAM
uv pip install -e .          # recommended
# or: pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, CUDA GPU recommended.

---

## Training

### Overfit on a single clip (sanity check)

```bash
python scripts/train.py \
  --dataset moma \
  --moma_data_root /path/to/moma_dataset \
  --max_samples 1 \
  --epochs 5000 \
  --batch_size 1 \
  --lr 1e-4 \
  --vis_every 100
```

### Full training run

```bash
python scripts/train.py \
  --dataset moma \
  --moma_data_root /path/to/moma_dataset \
  --batch_size 4 \
  --epochs 100 \
  --lr 3e-4 \
  --moma_num_frames 5 \
  --num_predict_frames 5 \
  --ckpt_dir checkpoints/run1 \
  --tensorboard_dir runs/run1
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--moma_num_frames` | 5 | Context frames fed to the model |
| `--num_predict_frames` | 5 | Future frames the DiT world model predicts |
| `--action_chunk_size` | 16 | Timesteps per predicted action chunk |
| `--action_dim` | 32 | Proprioceptive action dimensionality |
| `--moma_frame_stride` | 25 | Stride between sampled video frames |
| `--visual_loss_weight` | — | Weight on DiT frame-prediction loss |
| `--vis_every` | — | Save visualisation GIFs every N steps |

### Visualisation at inference

```bash
python scripts/visualize.py \
  --checkpoint checkpoints/run1/model_best.pt \
  --moma_data_root /path/to/moma_dataset \
  --output_dir vis_output \
  --diffusion_steps 50
```

---

## Model configuration

All hyperparameters live in a single dataclass — `HaloVLMConfig` in `config/model_config.py`.

```python
from config import HaloVLMConfig

config = HaloVLMConfig(
    emb_dim=512,
    dec_num_layers=8,
    dec_num_heads=16,
    use_moe=True,
    moe_num_routed_experts=4,
    moe_top_k=2,
    action_chunk_size=16,
    flow_num_ode_steps=24,
    dit_hidden_size=512,
    dit_depth=8,
    dit_num_sample_steps=50,
    num_visual_predict_frames=5,
)
```

---

## What the visualisations show

**Training GIF** — saved periodically during training, each GIF walks through three phases:

1. **Context scroll** — the N input camera frames the model actually sees
2. **Text reveal** — the model's generated description, token by token
3. **Action trajectory** — GT (solid) vs predicted (dashed) action dimensions, animated step by step
4. **Future frames** — GT future frame alongside the DiT's predicted future frame

**Diffusion GIF** — shows the DiT world model's denoising process for a single future frame: starting from pure Gaussian noise at `t=0` and integrating forward step by step to `t=1`, where the final frame is the model's prediction of the next scene.

---

## License

MIT — see [LICENSE](LICENSE).
