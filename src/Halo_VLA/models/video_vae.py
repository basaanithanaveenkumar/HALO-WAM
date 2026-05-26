"""
Video VAE: SD-style convolutional autoencoder with 8× spatial compression.

Compresses RGB frames [B, 3, H, W] → latents [B, 4, H/8, W/8].
Architecture mirrors Stable Diffusion VAE (f=8):

  Encoder  channels [128, 256, 512, 512]  3 × 2× strided-conv downsampling
  Decoder  channels [512, 512, 256, 128]  3 × nearest+conv upsampling
  Bottleneck (at 1/8 resolution): ResBlock → Self-Attention → ResBlock
  Latent: 4-channel diagonal Gaussian (encoder outputs 8 ch = mean ‖ logvar)

Latent scaling: latents are multiplied by SCALE_FACTOR = 0.18215 (SD
convention) so the DiT sees approximately unit-variance inputs.

Why VAE + DiT instead of pixel-space DiT?
  At 224×224 with patch_size=4 the pixel-space DiT sees 56×56 = 3 136 tokens.
  After 8× VAE compression the latent is 28×28 with patch_size=4 → 49 tokens.
  Same compute budget → 64× fewer attention operations → higher quality.

Training losses (returned by VideoVAE.forward):
  L_recon  MSE( decode(encode(x)), x )      pixel reconstruction
  L_kl     KL( q(z|x) || N(0,I) )           posterior regularisation

Usage:
    vae = VideoVAE()

    # Training
    out = vae(frame)                          # VAEOutput
    loss = F.mse_loss(out.x_recon, frame) + 1e-4 * out.kl_loss

    # Encode for DiT (returns scaled latent)
    z = vae.encode(frame)                     # [B, 4, H/8, W/8]

    # Decode DiT output back to pixels
    frame = vae.decode(z)                     # [B, 3, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-activation residual block: GroupNorm → SiLU → Conv, ×2, + skip."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch,  eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch,  out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Spatial self-attention (single-head, 1×1 conv QKV) at the bottleneck."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm  = nn.GroupNorm(32, ch, eps=1e-6, affine=True)
        self.q     = nn.Conv2d(ch, ch, 1)
        self.k     = nn.Conv2d(ch, ch, 1)
        self.v     = nn.Conv2d(ch, ch, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        q   = self.q(h).view(B, C, H * W).permute(0, 2, 1)   # [B, HW, C]
        k   = self.k(h).view(B, C, H * W)                     # [B, C,  HW]
        v   = self.v(h).view(B, C, H * W).permute(0, 2, 1)   # [B, HW, C]
        att = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(att, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(out)


class MidBlock(nn.Module):
    """Bottleneck: ResBlock → Attention → ResBlock."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.res1 = ResBlock(ch, ch)
        self.attn = AttentionBlock(ch)
        self.res2 = ResBlock(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.attn(self.res1(x)))


class Downsample(nn.Module):
    """2× spatial downsampling via strided convolution."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2× spatial upsampling via nearest-neighbour interpolation + convolution."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# Encoder / Decoder stage blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    num_res ResBlocks (first handles channel transition) + optional Downsample.

    E.g. EncoderBlock(128, 256, num_res=2, downsample=True):
        ResBlock(128→256) → ResBlock(256→256) → Downsample
    """

    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2, downsample: bool = True) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResBlock(in_ch, out_ch)]
        blocks += [ResBlock(out_ch, out_ch) for _ in range(num_res - 1)]
        self.resblocks = nn.Sequential(*blocks)
        self.down      = Downsample(out_ch) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.resblocks(x))


class DecoderBlock(nn.Module):
    """
    num_res ResBlocks (first handles channel transition) + optional Upsample.

    E.g. DecoderBlock(512, 256, num_res=3, upsample=True):
        ResBlock(512→256) → ResBlock(256→256) → ResBlock(256→256) → Upsample
    """

    def __init__(self, in_ch: int, out_ch: int, num_res: int = 3, upsample: bool = True) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResBlock(in_ch, out_ch)]
        blocks += [ResBlock(out_ch, out_ch) for _ in range(num_res - 1)]
        self.resblocks = nn.Sequential(*blocks)
        self.up        = Upsample(out_ch) if upsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.resblocks(x))


# ---------------------------------------------------------------------------
# VAE Encoder
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """
    [B, 3, H, W] → [B, latent_ch*2, H/8, W/8]   (mean ‖ logvar concatenated)

    Channel schedule : [128, 256, 512, 512]
    Downsampling     : ×2 after blocks 0, 1, 2  →  8× total
    Bottleneck       : ResBlock + Attention + ResBlock at 1/8 resolution
    """

    def __init__(self, in_ch: int = 3, latent_ch: int = 4, base_ch: int = 128) -> None:
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]

        self.conv_in = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        self.blocks = nn.ModuleList([
            EncoderBlock(ch[0], ch[0], num_res=2, downsample=True),   # H   → H/2
            EncoderBlock(ch[0], ch[1], num_res=2, downsample=True),   # H/2 → H/4
            EncoderBlock(ch[1], ch[2], num_res=2, downsample=True),   # H/4 → H/8
            EncoderBlock(ch[2], ch[3], num_res=2, downsample=False),  # H/8 (bottleneck)
        ])

        self.mid      = MidBlock(ch[-1])
        self.norm_out = nn.GroupNorm(32, ch[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(ch[-1], latent_ch * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.mid(h)
        return self.conv_out(F.silu(self.norm_out(h)))  # [B, latent_ch*2, H/8, W/8]


# ---------------------------------------------------------------------------
# VAE Decoder
# ---------------------------------------------------------------------------

class VAEDecoder(nn.Module):
    """
    [B, latent_ch, H/8, W/8] → [B, 3, H, W]

    Channel schedule : [512, 512, 256, 128]  (mirror of encoder)
    Upsampling       : ×2 after blocks 0, 1, 2  →  8× total
    Bottleneck       : ResBlock + Attention + ResBlock at 1/8 resolution
    """

    def __init__(self, latent_ch: int = 4, out_ch: int = 3, base_ch: int = 128) -> None:
        super().__init__()
        ch = [base_ch * 4, base_ch * 4, base_ch * 2, base_ch]

        self.conv_in = nn.Conv2d(latent_ch, ch[0], 3, padding=1)
        self.mid     = MidBlock(ch[0])

        self.blocks = nn.ModuleList([
            DecoderBlock(ch[0], ch[1], num_res=3, upsample=True),    # H/8 → H/4
            DecoderBlock(ch[1], ch[2], num_res=3, upsample=True),    # H/4 → H/2
            DecoderBlock(ch[2], ch[3], num_res=3, upsample=True),    # H/2 → H
            DecoderBlock(ch[3], ch[3], num_res=3, upsample=False),   # H   (final refine)
        ])

        self.norm_out = nn.GroupNorm(32, ch[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(ch[-1], out_ch, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.conv_in(z))
        for block in self.blocks:
            h = block(h)
        return self.conv_out(F.silu(self.norm_out(h)))  # [B, out_ch, H, W]


# ---------------------------------------------------------------------------
# VideoVAE
# ---------------------------------------------------------------------------

@dataclass
class VAEOutput:
    """All outputs from a VideoVAE forward pass (used during joint training)."""
    x_recon: torch.Tensor   # [B, 3, H, W]        pixel reconstruction
    z:       torch.Tensor   # [B, 4, H/8, W/8]    scaled latent (×SCALE_FACTOR)
    mean:    torch.Tensor   # [B, 4, H/8, W/8]    posterior mean
    logvar:  torch.Tensor   # [B, 4, H/8, W/8]    posterior log-variance (clamped)
    kl_loss: torch.Tensor   # scalar               KL( q(z|x) || N(0,I) )


class VideoVAE(nn.Module):
    """
    SD-style Video VAE — 8× spatial compression for robot video frames.

    Token count comparison for 224×224 frames with DiT patch_size=4:
      Pixel-space DiT : 56 × 56 = 3 136 tokens
      VAE latent DiT  : 28 × 28 = 784 pixels → 7 × 7 = 49 tokens   (64× fewer)

    The same DiT compute budget produces dramatically higher quality when
    the model diffuses in a compact latent space instead of raw pixels.

    Parameter count at base_ch=128 : ~66 M  (encoder ~30 M, decoder ~36 M)
    Use base_ch=64 for a lighter    : ~17 M  variant.

    Typical workflow
    ----------------
    Joint training (VAE + DiT trained together):

        # --- VAE loss ---
        vae_out  = vae(frame)
        vae_loss = F.mse_loss(vae_out.x_recon, frame) + kl_w * vae_out.kl_loss

        # --- DiT operates on stop-gradient latents for stability ---
        z_sg     = vae_out.z.detach()
        dit_loss = cfm(dit, z_sg, conditioning)

        loss = dit_loss + vae_loss

    Inference:
        z     = vae.encode(past_frame, sample=False)   # deterministic
        z_gen = euler_integrate(dit, conditioning)     # DiT generates latent
        frame = vae.decode(z_gen)                      # back to pixels
    """

    SCALE_FACTOR: float = 0.18215   # keeps latent std ≈ 1 (same as SD)
    LATENT_CH:    int   = 4

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128) -> None:
        super().__init__()
        self.encoder = VAEEncoder(in_ch=in_ch,              latent_ch=self.LATENT_CH, base_ch=base_ch)
        self.decoder = VAEDecoder(latent_ch=self.LATENT_CH, out_ch=out_ch,            base_ch=base_ch)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split 8-channel encoder output into mean and clamped logvar."""
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mean + std * ε,  ε ~ N(0,I)."""
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

    @staticmethod
    def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(mean, exp(logvar)) ‖ N(0,I) ) averaged over all elements."""
        return (-0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())).mean()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode RGB pixels to scaled latents.

        Args:
            x:      [B, 3, H, W]   input frames in model normalisation range.
            sample: True  → sample  z ~ q(z|x)   (training / stochastic)
                    False → use posterior mean    (deterministic inference)
        Returns:
            [B, 4, H/8, W/8] scaled by SCALE_FACTOR.
        """
        h              = self.encoder(x)
        mean, logvar   = self._split(h)
        z              = self.reparameterize(mean, logvar) if sample else mean
        return z * self.SCALE_FACTOR

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode scaled latents back to RGB pixels.

        Args:
            z: [B, 4, H/8, W/8]  scaled by SCALE_FACTOR.
        Returns:
            [B, 3, H, W] reconstructed / generated frames.
        """
        return self.decoder(z / self.SCALE_FACTOR)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """
        Full encode → reparameterise → decode pass (for VAE training).

        Args:
            x: [B, 3, H, W] input frames.
        Returns:
            VAEOutput containing reconstruction, latent, and KL loss.
        """
        h              = self.encoder(x)
        mean, logvar   = self._split(h)
        z              = self.reparameterize(mean, logvar)
        x_recon        = self.decoder(z)
        return VAEOutput(
            x_recon = x_recon,
            z       = z * self.SCALE_FACTOR,
            mean    = mean,
            logvar  = logvar,
            kl_loss = self.kl_divergence(mean, logvar),
        )
