"""
Model configuration for Halo-VLA.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.tokens import get_vocab_size, get_token_id


@dataclass
class HaloVLMConfig:
    """All hyper-parameters for HaloVLM in one place."""

    # Tokenizer / vocabulary (auto-set from special_tokens.json)
    # Using HuggingFaceTB/cosmo2-tokenizer (SmolLM tokenizer, 49152 base vocab)
    # which uses the same ChatML format as Qwen2.5 but is 3× smaller.
    vocab_size: int = 49156  # cosmo2-tokenizer (49152) + 4 custom tokens

    # Special token IDs (from config/special_tokens.json)
    image_token_id: int = 49152
    action_token_id: int = 49153
    state_token_id: int = 49154
    world_video_token_id: int = 49155

    # Shared embedding dimension
    emb_dim: int = 512

    # Factored-embedding bottleneck (ALBERT-style).
    # token_emb: vocab × emb_factor_dim   (shared with lm_head)
    # emb_proj:  emb_factor_dim → emb_dim (input side)
    # lm_proj:   emb_dim → emb_factor_dim (output side)
    # With emb_factor_dim=128: ~19.5M params vs ~155M for a full table.
    emb_factor_dim: int = 128

    # Vision encoder (ViT)
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    vit_num_layers: int = 6
    vit_num_heads: int = 16
    vit_mlp_dim: int = 512
    vit_drop: float = 0.0

    # Decoder transformer
    dec_num_layers: int = 8
    dec_num_heads: int = 16
    dec_mlp_dim: int = 512
    dec_drop: float = 0.0

    # MoE (DeepseekMoE) — used inside each TransformerBlock
    use_moe: bool = True                  # False → standard MLP FFN
    moe_hid_scale: float = 1.2            # hidden dim = round(emb_dim * scale)
    moe_num_routed_experts: int = 6
    moe_top_k: int = 2
    moe_num_shared_experts: int = 2

    # Positional embeddings
    max_position_embeddings: int = 2000

    # Image projector
    proj_vision_dim: int | None = None   # defaults to emb_dim
    proj_llm_dim: int | None = None      # defaults to emb_dim

    # Action decoder
    action_dim: int = 7                  # output dims (e.g. 6-DOF + gripper)
    action_hidden_dims: tuple[int, ...] = (512, 256)
    action_chunk_size: int = 16          # predict N future steps at once
    action_dropout: float = 0.1
    action_use_layernorm: bool = True

    # State encoder
    state_dim: int = 32                  # raw proprioceptive state dims
    state_hidden_dims: tuple[int, ...] = (256, 512)
    state_dropout: float = 0.1
    state_use_layernorm: bool = True

    # Flow matching action decoder
    flow_hidden_dim: int = 1024          # hidden dim in flow velocity network
    flow_time_embed_dim: int = 128       # dim of sinusoidal time embedding
    flow_num_ode_steps: int = 24        # Euler integration steps at inference

    # DiT visual heads (frame / depth / flow) — used by models.dit_frame_prediction
    dit_in_channels: int = 3               # default RGB; overridden per head when needed
    dit_out_channels: int = 3                # velocity has same layout as x (flow matching)
    
    # Visual prediction channels
    rgb_channels: int = 3
    depth_channels: int = 1
    flow_channels: int = 2

    dit_time_freq_dim: int = 512      # sinusoidal timestep embedding size — matches dit_hidden_size
    dit_patch_size: int = 8            # 224/8 = 28×28 = 784 tokens (4× finer than patch=16)
    dit_hidden_size: int = 512          # matches emb_dim — no context compression loss
    dit_depth: int = 8                  # twice the velocity-field capacity
    dit_num_heads: int = 8              # hidden/heads = 64 (same ratio as before)
    dit_mlp_ratio: float = 4.0
    dit_max_resolution: int = 224        # pixel resolution of the DiT input (same as img_size)
    dit_rgb_in_channels: int = 3         # convenience mirrors for make_dit_config
    dit_depth_in_channels: int = 1
    dit_flow_in_channels: int = 2
    dit_flow_smooth_weight: float = 0.1
    dit_depth_recon_weight: float = 1.0
    dit_flow_photo_weight: float = 0.5
    dit_num_sample_steps: int = 75      # more Euler steps at inference → better quality

    # Gradient checkpointing — recomputes activations on backward instead of storing them.
    # Saves ~40% activation memory on the decoder at ~33% extra compute cost.
    gradient_checkpointing: bool = True

    # HaloVLM integration: DiT RGB / depth / flow heads (``VisualDiTPredictor``)
    enable_visual_dit: bool = True      # False → omit heads (e.g. old checkpoints)
    visual_loss_weight: float = 8.0      # increased to push more training signal into DiT head
    # Number of future frames the DiT head is asked to predict (configurable).
    # The dataloader samples num_sample_frames (context) + num_visual_predict_frames
    # (targets) and returns them as "images" and "future_frames" respectively.
    num_visual_predict_frames: int = 5   # focus capacity on near-future frames

    # System prompt
    system_prompt: str = (
        "You are a robotic VLA assistant. Given images and states, "
        "describe observations or output <halo_action> with a predicted trajectory."
    )
