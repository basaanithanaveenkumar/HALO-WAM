import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional-style embedding for a continuous scalar t ∈ [0, 1]."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Pre-compute log-spaced frequencies (not learnable)
        half = embed_dim // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / half))
        self.register_buffer("freq", freq)  # (half,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) or scalar → (batch, embed_dim)"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # t: (batch,) → (batch, 1)
        t = t.unsqueeze(-1)  # (batch, 1)
        args = t * self.freq  # (batch, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, embed_dim)


class FlowActionDecoder(nn.Module):
    """
    Predicts the vector field v_theta(x_t, t, condition).
    Inputs:
        x_t : (batch, action_chunk_size)  – noisy action chunk at time t (flattened)
        t   : (batch,) or scalar          – current time in [0,1]
        cond: (batch, obs_dim)            – observation / condition
    Output:
        v   : (batch, action_chunk_size)  – predicted vector field
    """
    def __init__(self, action_dim_flat, obs_dim, hidden_dim=1024, time_embed_dim=128):
        super().__init__()
        self.action_dim_flat = action_dim_flat
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Sinusoidal time embedding: scalar t → (batch, time_embed_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Input: concatenate [x_t, time_emb, cond] → total dimension
        self.net = nn.Sequential(
            nn.Linear(action_dim_flat + time_embed_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim_flat)
        )

    def forward(self, x_t, t, cond):
        # Sinusoidal embedding of t: (batch,) → (batch, time_embed_dim)
        t_emb = self.time_embed(t)
        # Concatenate along feature dimension
        inp = torch.cat([x_t, t_emb, cond], dim=1)  # (batch, action_dim_flat+time_embed_dim+obs_dim)
        return self.net(inp)