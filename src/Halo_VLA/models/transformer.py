import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.moe import DeepseekMoE


class SelfAttn(nn.Module):
    def __init__(self, emb_dim, return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim

        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.val_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
        self.return_attn_weights = return_attn_weights

    def forward(self, inputs, causal_mask=None, pad_mask=None):
        # inputs: [B, Seq_len, D]
        Query = self.query_proj(inputs)  # [B, Seq_len, D]
        Key = self.key_proj(inputs)      # [B, Seq_len, D]
        Val = self.val_proj(inputs)      # [B, Seq_len, D]

        attn_scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale  # [B, Seq_len, Seq_len]

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # pad_mask: [B, Seq_len] with 1=real, 0=pad; mask out pad key positions
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(pad_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Seq_len, Seq_len]
        attn_output = torch.matmul(attn_weights, Val)   # [B, Seq_len, D]

        if self.return_attn_weights:
            return attn_output, attn_weights
        return attn_output


class HeadAttn(nn.Module):
    def __init__(self, emb_dim=256, head_size=16, drop_fact=0.0, causal_mask=False, return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim

        self.query_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.key_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.val_proj = nn.Linear(emb_dim, head_size, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
        self.causal_mask = causal_mask
        self.dropout = nn.Dropout(drop_fact)
        self.return_attn_weights = return_attn_weights

    def forward(self, inputs, pad_mask=None):
        # inputs: [B, Seq_len, D]
        B, seq_len, D = inputs.shape
        Query = self.query_proj(inputs)  # [B, Seq_len, head_size]
        Key = self.key_proj(inputs)      # [B, Seq_len, head_size]
        Val = self.val_proj(inputs)      # [B, Seq_len, head_size]

        scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale  # [B, Seq_len, Seq_len]

        if self.causal_mask:
            causal_tril = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs.device))
            scores = scores.masked_fill(causal_tril == 0, float('-inf'))

        # pad_mask: [B, Seq_len] with 1=real, 0=pad; mask out pad key positions
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, Seq_len, Seq_len]
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, Val)  # [B, Seq_len, head_size]

        if self.return_attn_weights:
            return attn_output, attn_weights
        return attn_output


class MultiHeadAttn(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, drop_fact=0.0, causal_mask=False, return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim
        self.num_heads = num_heads
        assert emb_dim % num_heads == 0, f"emb_dim {emb_dim} must be divisible by num_heads {num_heads}"

        self.head_size = emb_dim // num_heads
        self.causal_mask = causal_mask
        self.heads = nn.ModuleList([
            HeadAttn(emb_dim, head_size=self.head_size, drop_fact=drop_fact,
                     causal_mask=causal_mask, return_attn_weights=return_attn_weights)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_fact)

    def forward(self, inputs, pad_mask=None):
        B, L, D = inputs.shape
        # Project all heads at once → [B, H, L, head_size].
        # Weights come from the existing per-head Linear layers (checkpoint-compatible).
        Q = torch.stack([h.query_proj(inputs) for h in self.heads], dim=1)
        K = torch.stack([h.key_proj(inputs)   for h in self.heads], dim=1)
        V = torch.stack([h.val_proj(inputs)   for h in self.heads], dim=1)

        # Boolean attend-mask (True = this position is allowed to be attended to).
        attn_mask = None
        if self.causal_mask:
            # Lower triangle: token i can attend to tokens j <= i.
            causal = torch.tril(torch.ones(L, L, device=inputs.device, dtype=torch.bool))
            attn_mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        if pad_mask is not None:
            # pad_mask [B, L]: 1=real, 0=pad.  Mask key positions that are padding.
            key_mask = pad_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = (attn_mask & key_mask) if attn_mask is not None else key_mask

        # F.scaled_dot_product_attention uses Flash Attention when available,
        # avoiding the explicit [B, H, L, L] attention matrix entirely.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # [B, H, L, head_size]

        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        return self.dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, mlp_dim=512, drop_fact=0.0,
                 causal_mask=False, use_moe=True, moe_hid_scale=1.2,
                 moe_num_routed_experts=16, moe_top_k=4,
                 moe_num_shared_experts=2):
        super().__init__()
        self.attn = MultiHeadAttn(emb_dim=emb_dim, num_heads=num_heads, drop_fact=drop_fact, causal_mask=causal_mask)
        self.norm1 = nn.LayerNorm(emb_dim)

        self.use_moe = use_moe
        if use_moe:
            moe_hid_dim = round(emb_dim * moe_hid_scale)
            self.ffn = DeepseekMoE(
                emb_dim, moe_hid_dim,
                num_router_exprts=moe_num_routed_experts,
                best_k=moe_top_k,
                num_shared_exprts=moe_num_shared_experts,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, emb_dim),
                nn.Dropout(drop_fact),
            )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, inputs, pad_mask=None):
        attn_output = self.attn(self.norm1(inputs), pad_mask=pad_mask)
        x = inputs + attn_output
        ffn_output = self.ffn(self.norm2(x))
        return x + ffn_output


class DecoderTransformer(nn.Module):
    def __init__(self, num_layers=16, emb_dim=1024, num_heads=32, mlp_dim=512,
                 drop_fact=0.0, use_moe=True, moe_hid_scale=1.2,
                 moe_num_routed_experts=16, moe_top_k=4,
                 moe_num_shared_experts=2, gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                drop_fact=drop_fact, causal_mask=True,
                use_moe=use_moe,
                moe_hid_scale=moe_hid_scale,
                moe_num_routed_experts=moe_num_routed_experts,
                moe_top_k=moe_top_k,
                moe_num_shared_experts=moe_num_shared_experts,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, inputs, pad_mask=None):
        from torch.utils.checkpoint import checkpoint
        x = inputs
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Recompute activations during backward instead of storing them.
                # Saves ~activation memory per layer at the cost of one extra forward pass.
                x = checkpoint(layer, x, pad_mask, use_reentrant=False)
            else:
                x = layer(x, pad_mask=pad_mask)
        return self.norm(x)
