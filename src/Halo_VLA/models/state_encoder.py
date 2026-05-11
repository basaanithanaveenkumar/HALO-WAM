"""
State Encoder for Halo-VLA.

MLP-based module that projects raw robot proprioceptive state vectors
into the transformer's embedding space so they can be injected as
tokens alongside image and text embeddings.

Expected input:  raw state vector  [B, state_dim]  (e.g. 32-dim)
Expected output: state embedding   [B, emb_dim]    (matches transformer dim)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from config import HaloVLMConfig


class StateEncoder(nn.Module):
    """
    Configurable MLP state encoder.

    Maps raw proprioceptive state (joint positions, gripper state, etc.)
    into the model's embedding space.

    Args:
        config: HaloVLMConfig (uses state_dim, emb_dim, state_hidden_dims,
                state_dropout, state_use_layernorm).
        Alternatively, pass keyword overrides.
    """

    def __init__(self, config: HaloVLMConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = HaloVLMConfig(**kwargs)
        self.config = config

        in_dim: int = config.state_dim
        hidden_dims: Sequence[int] = config.state_hidden_dims
        out_dim: int = config.emb_dim
        dropout: float = config.state_dropout
        use_ln: bool = config.state_use_layernorm

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_ln:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection → embedding space
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Tensor of shape [B, state_dim]
                   Raw proprioceptive state vector.

        Returns:
            state_emb: Tensor of shape [B, emb_dim]
                       Embedding suitable for injection into the
                       transformer sequence as a state token.
        """
        return self.mlp(state)  # [B, emb_dim]


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = HaloVLMConfig(
        state_dim=32,
        emb_dim=512,
        state_hidden_dims=(256, 512),
        state_dropout=0.1,
        state_use_layernorm=True,
    )
    encoder = StateEncoder(config=cfg)
    print(encoder)

    x = torch.randn(2, cfg.state_dim)
    out = encoder(x)
    print(f"\nInput:  {x.shape}  (raw state)")
    print(f"Output: {out.shape}  (expected [2, {cfg.emb_dim}])")
