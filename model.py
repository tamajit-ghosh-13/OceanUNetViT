"""
================================================================================
OceanEmbed - Model Architecture (model.py)  [UPGRADED v2]
================================================================================
CHANGES FROM v1:
  - in_channels: 3 → 7 (SST, SSS, SSH, U_CUR, V_CUR, U_WIND, V_WIND)
  - out_depth_levels: 14 → 15 (standard hackathon depth levels)
  - base_filters: 32 → 64 (more capacity for 7 input channels)
  - ViT heads: 4 → 8 (richer attention for larger bottleneck)
  - Added: get_embedding() method for latent space extraction
  - Added: 4-level encoder/decoder (deeper U-Net for 101×241 spatial input)

ARCHITECTURE SUMMARY:
  [Input: (B, 7, 101, 241)]
       ↓
  ┌────────────────────────────┐
  │  ENCODER (4 levels, CNN)   │  Progressively compresses spatial resolution
  │  Level 1: (B, 64, 101, 241)│  Full resolution — fine coastal details
  │  Level 2: (B, 128, 50, 120)│  Half resolution — mesoscale eddies
  │  Level 3: (B, 256, 25, 60) │  Quarter — regional circulation patterns
  │  Level 4: (B, 512, 12, 30) │  Eighth — basin-scale dynamics
  └─────────────┬──────────────┘
               ↓
  ┌────────────────────────────┐
  │  BOTTLENECK (ViT, 8 heads) │  Global North Indian Ocean teleconnections
  │  Compact Embedding Vector  │  (B, 256) — extracted by get_embedding()
  └─────────────┬──────────────┘
               ↓
  ┌────────────────────────────┐
  │  DECODER (4 levels)        │  Upsamples + skip connections
  │  Projects to 15 depth layers│
  └────────────────────────────┘
       ↓
  [Output: (B, 15, 101, 241)]
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from config import N_INPUT_CHANNELS, N_DEPTH_LEVELS, MODEL


# ==============================================================================
# 1. Double Convolution Block (core U-Net building block)
# ==============================================================================
class DoubleConv(nn.Module):
    """
    Two consecutive convolution operations with normalization and activation.
    Pattern: Conv2d → BatchNorm → GELU → Conv2d → BatchNorm → GELU

    Why GELU over ReLU? GELU (Gaussian Error Linear Unit) is smoother and works
    better with Vision Transformers, making it ideal for our hybrid architecture.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ==============================================================================
# 2. Encoder Block (Downsampling Step)
# ==============================================================================
class EncoderBlock(nn.Module):
    """
    One encoder level: applies DoubleConv, then MaxPool to halve spatial dimensions.
    Stores the pre-pool output as a 'skip connection' for the decoder.

    Think of this like zooming out on a map to see broader patterns.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          skip : tensor before pooling (for skip connection to decoder)
          pooled: tensor after pooling (input to the next encoder level)
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


# ==============================================================================
# 3. Decoder Block (Upsampling Step + Skip Connection)
# ==============================================================================
class DecoderBlock(nn.Module):
    """
    One decoder level: upsamples, concatenates skip connection from encoder,
    then applies DoubleConv to merge and refine spatial features.

    The skip connections preserve fine-grained coastal/eddy details that
    would otherwise be lost during downsampling.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions: pad if necessary
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return self.conv(x)


# ==============================================================================
# 4. Vision Transformer Bottleneck
# ==============================================================================
class TransformerBottleneck(nn.Module):
    """
    Self-Attention bottleneck at the lowest spatial resolution.

    At the bottleneck (12 × 30 = 360 tokens), each token represents a
    roughly 2° × 2° patch of the North Indian Ocean. Self-attention lets
    the Arabian Sea "talk to" the Bay of Bengal directly — capturing the
    basin-scale teleconnections that CNNs cannot model with local filters.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims → sequence of tokens: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        # Self-Attention with residual
        normed = self.norm1(tokens)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=True)
        tokens = tokens + self.dropout(attn_out)
        # Feed-Forward MLP with residual
        tokens = tokens + self.mlp(self.norm2(tokens))
        # Reshape back to spatial map: (B, C, H, W)
        out_spatial = tokens.transpose(1, 2).reshape(B, C, H, W)
        if return_attention:
            return out_spatial, attn_weights
        return out_spatial


# ==============================================================================
# 5. Compact Embedding Projector (Latent Space)
# ==============================================================================
class EmbeddingProjector(nn.Module):
    """
    Extracts a compact fixed-size embedding vector from the bottleneck features.

    This satisfies the hackathon requirement for a "compact satellite embedding
    that captures hidden ocean dynamics."

    Projects (B, C, H, W) → (B, embedding_dim) using global average pooling
    followed by a linear projection layer.
    """
    def __init__(self, in_channels: int, embedding_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling → (B, C, 1, 1)
        self.proj = nn.Sequential(
            nn.Flatten(),                         # (B, C)
            nn.Linear(in_channels, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(x))


# ==============================================================================
# 6. Complete Model: OceanUNetViT
# ==============================================================================
class OceanUNetViT(nn.Module):
    """
    Hybrid U-Net + Vision Transformer for 3D Ocean Temperature Field Inversion.

    Input:  (Batch, 7, 101, 241) — 7 daily surface satellite channels
    Output: (Batch, 15, 101, 241) — temperature at 15 standard depth levels (0m–1000m)

    Embedding: (Batch, 256) — compact latent representation (via get_embedding())
    """
    def __init__(
        self,
        in_channels: int = MODEL["in_channels"],
        out_depth_levels: int = MODEL["out_depth_levels"],
        base_filters: int = MODEL["base_filters"],
        vit_heads: int = MODEL["vit_heads"],
        vit_mlp_ratio: float = MODEL["vit_mlp_ratio"],
        embedding_dim: int = MODEL["embedding_dim"],
    ):
        super().__init__()
        f = base_filters  # shorthand: 64

        # ----------------------------------------------------------------------
        # ENCODER — 4 levels of downsampling
        # ----------------------------------------------------------------------
        # Level 1:  in=(7)        → out=(f=64)    | spatial: 101×241
        self.enc1 = EncoderBlock(in_channels, f)
        # Level 2:  in=(f=64)     → out=(2f=128)  | spatial: 50×120
        self.enc2 = EncoderBlock(f, f * 2)
        # Level 3:  in=(2f=128)   → out=(4f=256)  | spatial: 25×60
        self.enc3 = EncoderBlock(f * 2, f * 4)
        # Level 4:  in=(4f=256)   → out=(8f=512)  | spatial: 12×30
        self.enc4 = EncoderBlock(f * 4, f * 8)

        # ----------------------------------------------------------------------
        # BOTTLENECK — Vision Transformer at lowest resolution (12×30)
        # ----------------------------------------------------------------------
        bottleneck_ch = f * 8  # 512
        self.bottleneck_conv = DoubleConv(bottleneck_ch, bottleneck_ch)
        self.vit = TransformerBottleneck(
            embed_dim=bottleneck_ch,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
        )

        # ----------------------------------------------------------------------
        # EMBEDDING PROJECTOR — Compact latent representation
        # ----------------------------------------------------------------------
        self.embedding_proj = EmbeddingProjector(bottleneck_ch, embedding_dim)

        # ----------------------------------------------------------------------
        # DECODER — 4 levels of upsampling with skip connections
        # ----------------------------------------------------------------------
        # Up 4→3:  bottleneck(512) up→256, concat s4(512 from enc4) → DoubleConv(768→256)
        self.dec4 = DecoderBlock(bottleneck_ch, f * 8, f * 4)
        # Up 3→2:  dec4_out(256) up→128, concat s3(256 from enc3) → DoubleConv(384→128)
        self.dec3 = DecoderBlock(f * 4, f * 4, f * 2)
        # Up 2→1:  dec3_out(128) up→64,  concat s2(128 from enc2) → DoubleConv(192→64)
        self.dec2 = DecoderBlock(f * 2, f * 2, f)
        # Up 1→0:  dec2_out(64)  up→32,  concat s1(64 from enc1)  → DoubleConv(96→64)
        self.dec1 = DecoderBlock(f, f, f)

        # ----------------------------------------------------------------------
        # OUTPUT HEAD — Projects from base_filters → 15 depth levels
        # ----------------------------------------------------------------------
        self.output_head = nn.Conv2d(f, out_depth_levels, kernel_size=1)
        # Learnable per-depth bias correction layer (Feature 4)
        self.depth_bias = nn.Parameter(torch.zeros(1, out_depth_levels, 1, 1))

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False, return_attention: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor — (B, in_channels, 101, 241)
        return_embedding : bool — if True, returns (predictions, embedding)
        return_attention : bool — if True, returns (predictions, attn_weights)

        Returns:
        --------
        predictions : (B, 15, 101, 241)
        embedding   : (B, 256)  — only if return_embedding=True
        attention   : (B, num_heads, tokens, tokens) — only if return_attention=True
        """
        # --- Encoder ---
        s1, p1 = self.enc1(x)    # s1: (B,64,101,241)  p1: (B,64,50,120)
        s2, p2 = self.enc2(p1)   # s2: (B,128,50,120)  p2: (B,128,25,60)
        s3, p3 = self.enc3(p2)   # s3: (B,256,25,60)   p3: (B,256,12,30)
        s4, p4 = self.enc4(p3)   # s4: (B,512,12,30)   p4: (B,512,6,15)

        # --- Bottleneck ---
        b = self.bottleneck_conv(p4)   # (B, 512, 6, 15)
        if return_attention:
            b, attn_weights = self.vit(b, return_attention=True)
        else:
            b = self.vit(b, return_attention=False)

        # --- Compact Embedding ---
        embedding = self.embedding_proj(b)  # (B, 256)

        # --- Decoder with Skip Connections ---
        d4 = self.dec4(b, s4)    # (B, 256, 12, 30)
        d3 = self.dec3(d4, s3)   # (B, 128, 25, 60)
        d2 = self.dec2(d3, s2)   # (B, 64, 50, 120)
        d1 = self.dec1(d2, s1)   # (B, 64, 101, 241)

        # --- Output Projection + Depth Bias Correction ---
        predictions = self.output_head(d1) + self.depth_bias  # (B, 15, 101, 241)

        if return_embedding and return_attention:
            return predictions, embedding, attn_weights
        if return_embedding:
            return predictions, embedding
        if return_attention:
            return predictions, attn_weights
        return predictions

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts only the compact (B, 256) latent embedding without computing the decoder.
        Useful for transfer learning, visualization, or downstream tasks.
        """
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bottleneck_conv(p4)
        b = self.vit(b)
        return self.embedding_proj(b)


# ==============================================================================
# 7. Model Factory
# ==============================================================================
def create_model(
    in_channels: int = N_INPUT_CHANNELS,
    out_depth_levels: int = N_DEPTH_LEVELS,
) -> OceanUNetViT:
    """Instantiates the OceanUNetViT with project-standard configuration."""
    return OceanUNetViT(in_channels=in_channels, out_depth_levels=out_depth_levels)


# ==============================================================================
# 8. Self-Test
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Running model.py self-test (OceanUNetViT v2)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Device: {device}")

    model = create_model().to(device)

    # Test with correct spec dimensions: batch=2, 7 channels, 101×241
    dummy = torch.randn(2, N_INPUT_CHANNELS, 101, 241, device=device)

    with torch.no_grad():
        predictions, embedding = model(dummy, return_embedding=True)

    print(f"   Input:      {tuple(dummy.shape)} → (B, 7_channels, 101_lat, 241_lon)")
    print(f"   Output:     {tuple(predictions.shape)} → (B, 15_depths, 101_lat, 241_lon)")
    print(f"   Embedding:  {tuple(embedding.shape)} → (B, 256_latent_dim)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    assert predictions.shape == (2, N_DEPTH_LEVELS, 101, 241), "Output shape mismatch!"
    assert embedding.shape == (2, 256), "Embedding shape mismatch!"
    print("✅ model.py (v2) verified successfully!")
