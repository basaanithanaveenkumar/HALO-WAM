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

    # System prompt
    system_prompt: str = (
        "You are a robotic VLA assistant. Given images and states, "
        "describe observations or output <halo_action> with a predicted trajectory."
    )
