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
    vocab_size: int = 151668  # Qwen2.5 + custom tokens

    # Special token IDs (from config/special_tokens.json)
    image_token_id: int = 151665
    action_token_id: int = 151666
    state_token_id: int = 151667

    # Shared embedding dimension
    emb_dim: int = 512

    # Vision encoder (ViT)
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    vit_num_layers: int = 4
    vit_num_heads: int = 16
    vit_mlp_dim: int = 512
    vit_drop: float = 0.0

    # Decoder transformer
    dec_num_layers: int = 12
    dec_num_heads: int = 16
    dec_mlp_dim: int = 512
    dec_drop: float = 0.0

    # MoE (DeepseekMoE) — used inside each TransformerBlock
    use_moe: bool = True                  # False → standard MLP FFN
    moe_hid_scale: float = 1.2            # hidden dim = round(emb_dim * scale)
    moe_num_routed_experts: int = 8
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
    flow_num_ode_steps: int = 20         # Euler integration steps at inference

    # DiT visual heads (frame / depth / flow) — used by models.dit_frame_prediction
    dit_in_channels: int = 3               # default RGB; overridden per head when needed
    dit_out_channels: int = 3                # velocity has same layout as x (flow matching)
    dit_time_freq_dim: int = 256         # sinusoidal timestep embedding size before MLP
    dit_patch_size: int = 8
    dit_hidden_size: int = 256
    dit_depth: int = 6
    dit_num_heads: int = 8
    dit_mlp_ratio: float = 4.0
    dit_max_resolution: int = 32         # max tokens per side (covers 224÷8 = 28)
    dit_rgb_in_channels: int = 3           # convenience mirrors for make_dit_config
    dit_depth_in_channels: int = 1
    dit_flow_in_channels: int = 2
    dit_flow_smooth_weight: float = 0.1  # optical-flow edge-aware smoothness
    dit_depth_recon_weight: float = 0.5  # depth CFM + decoder RGB reconstruction
    dit_flow_photo_weight: float = 0.5  # photometric term on flow estimate
    dit_num_sample_steps: int = 20       # Euler steps for sampling at inference

    # HaloVLM integration: DiT RGB / depth / flow heads (``VisualDiTPredictor``)
    enable_visual_dit: bool = True      # False → omit heads (e.g. old checkpoints)
    visual_loss_weight: float = 0.2      # added to train.py total loss

    # System prompt
    system_prompt: str = (
        "You are a robotic VLA assistant. Given images and states, "
        "describe observations or output <halo_action> with a predicted trajectory."
    )
