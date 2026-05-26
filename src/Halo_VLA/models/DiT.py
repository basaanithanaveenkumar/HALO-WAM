import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Create sinusoidal embeddings from scalar timesteps t ∈ ℝ (e.g. flow time in [0, 1]).
    t: [B] — each batch element gets an independent embedding row of size dim.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ScalarTimestepEmbedder(nn.Module):
    """Maps scalar t [B] → vector [B, hidden_size] (DiT / flow-matching compatible)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.reshape(t.size(0))
        t_freq = _timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """A transformer block with adaptive LayerNorm (adaLN) modulation."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # Adaptive LN parameters: scale and shift for both norms, plus residual scales
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: [B, N, D]  token sequence
        c: [B, D]     conditioning vector (timestep + optional context)
        """
        # c is the adaLN conditioning vector (e.g., time embedding + context)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        # Modulate layer norm 1
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        # Self-attention with residual gating
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # Modulate layer norm 2
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class DiT(nn.Module):
    """Diffusion Transformer for image-like data (frames, depth, flow)."""
    def __init__(self, config):
        super().__init__()
        in_channels = config.dit_in_channels
        out_channels = config.dit_out_channels
        patch_size = config.dit_patch_size
        hidden_size = config.dit_hidden_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = config.dit_num_heads
        self.depth = config.dit_depth

        # Patchify: conv -> linear projection
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size,
            padding=0
        )
        # Number of visual tokens at maximum resolution (in patch units, not pixels)
        self.max_res = config.dit_max_resolution
        num_patches = (self.max_res // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=True
        )
        freq_dim = getattr(config, "dit_time_freq_dim", 256)
        self.time_embed = ScalarTimestepEmbedder(hidden_size, frequency_embedding_size=freq_dim)
        # Additional context embedding (e.g., from HaloVLM) will be fused
        self.context_embed = nn.Linear(config.emb_dim, hidden_size)  # assuming config.emb_dim from HaloVLM
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, self.num_heads, config.dit_mlp_ratio)
            for _ in range(self.depth)
        ])
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Unpatchify: linear -> conv to reconstruct pixel space
        self.unpatch_embed = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.out_channels = out_channels
        self.initialize_weights()

    def initialize_weights(self):
        # Standard weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Zero-out the last layer of each adaLN modulation for stability (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out the unpatch embedding output layer (optional)
        nn.init.constant_(self.unpatch_embed.weight, 0)

    def forward(self, x, t, context=None):
        """
        x:       [B, C, H, W]       noisy input (e.g., frame, depth, flow)
        t:       [B]                diffusion timestep (int or float)
        context: [B, D_context]     additional conditioning (from HaloVLM) or None
        Returns: [B, out_C, H, W]   predicted noise (or v-prediction)
        """
        B, C, H, W = x.shape
        # 1. Patchify
        x_patch = self.patch_embed(x)   # [B, hidden_size, H/patch, W/patch]
        ph, pw = x_patch.shape[2], x_patch.shape[3]
        x_patch = x_patch.flatten(2).transpose(1, 2)  # [B, N, hidden_size], N = ph*pw

        # 2. Add positional embedding (interpolate if resolution differs from max)
        max_ph = self.max_res // self.patch_size
        if ph != max_ph or pw != max_ph:
            pos_embed = F.interpolate(
                self.pos_embed.reshape(1, max_ph, max_ph, -1).permute(0, 3, 1, 2),
                size=(ph, pw), mode='bilinear', align_corners=False,
            ).flatten(2).permute(0, 2, 1)
        else:
            pos_embed = self.pos_embed
        x_patch = x_patch + pos_embed

        # 3. Prepare conditioning vector c = time_embed(t) + context_proj
        t_emb = self.time_embed(t)   # [B, hidden_size]
        if context is not None:
            ctx_emb = self.context_embed(context)  # [B, hidden_size]
            c = t_emb + ctx_emb
        else:
            c = t_emb

        # 4. Transformer blocks
        for block in self.blocks:
            x_patch = block(x_patch, c)

        # 5. Final LN and unpatchify
        x_patch = self.final_norm(x_patch)
        x_patch = self.unpatch_embed(x_patch)  # [B, N, patch*patch*out_C]
        # Reshape to spatial image
        x_out = x_patch.reshape(B, ph, pw, self.patch_size, self.patch_size, self.out_channels)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_out = x_out.reshape(B, self.out_channels, H, W)
        return x_out