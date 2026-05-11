"""
Action Decoder for Halo-VLA.

MLP-based head that maps transformer hidden states to continuous robot actions.
Supports configurable hidden layers, action chunking, and optional LayerNorm.

Expected input:  hidden states from the decoder transformer  [B, emb_dim]
Expected output: action predictions                         [B, action_chunk_size, action_dim]
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from config import HaloVLMConfig


class ActionDecoder(nn.Module):
    """
    Configurable MLP action decoder.

    Args:
        config: HaloVLMConfig (uses emb_dim, action_dim, action_hidden_dims,
                action_chunk_size, action_dropout, action_use_layernorm).
        Alternatively, pass keyword overrides.
    """

    def __init__(self, config: HaloVLMConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = HaloVLMConfig(**kwargs)
        self.config = config

        in_dim: int = config.emb_dim
        hidden_dims: Sequence[int] = config.action_hidden_dims
        out_dim: int = config.action_chunk_size * config.action_dim
        dropout: float = config.action_dropout
        use_ln: bool = config.action_use_layernorm

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_ln:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection → flat action vector
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

        self.action_dim = config.action_dim
        self.chunk_size = config.action_chunk_size

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: Tensor of shape [B, emb_dim]
                    (e.g. the last-token hidden state from the transformer,
                     or a pooled representation)

        Returns:
            actions: Tensor of shape [B, action_chunk_size, action_dim]
        """
        flat = self.mlp(hidden)                               # [B, chunk*dim]
        return flat.view(-1, self.chunk_size, self.action_dim)  # [B, T, D]


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = HaloVLMConfig(
        emb_dim=512,
        action_dim=7,
        action_hidden_dims=(512, 256),
        action_chunk_size=4,
        action_dropout=0.1,
        action_use_layernorm=True,
    )
    decoder = ActionDecoder(config=cfg)
    print(decoder)

    x = torch.randn(2, cfg.emb_dim)
    out = decoder(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {out.shape}  (expected [2, {cfg.action_chunk_size}, {cfg.action_dim}])")
