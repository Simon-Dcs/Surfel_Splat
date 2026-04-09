"""
Enhanced Depth Head with High Capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PixelShuffleUpsample(nn.Module):
    """
    PixelShuffle upsampling module with learnable upsampling.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        # PixelShuffle requires upscale_factor^2 times the number of channels
        shuffle_channels = out_channels * (upscale_factor ** 2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, shuffle_channels, 3, 1, 1),
            nn.GroupNorm(min(32, shuffle_channels // 4), shuffle_channels),
            nn.GELU(),
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Post-processing convolution
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(32, out_channels // 4), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.post_conv(x)
        return x


class EfficientSelfAttention2D(nn.Module):
    """
    Efficient 2D Self-Attention module with reduced memory usage.
    Uses chunked processing to avoid OOM.
    """
    def __init__(self, channels, num_heads=8, chunk_size=1024):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.chunk_size = chunk_size

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels // 4), channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Skip attention for large spatial resolutions to avoid OOM
        if N > 8192:
            return x

        # Layer norm
        x_norm = self.norm(x)

        # Generate Q, K, V
        qkv = self.qkv(x_norm)  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        # Attention with chunking to reduce memory
        scale = self.head_dim ** -0.5

        if N <= self.chunk_size:
            # Small enough, compute directly
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
            out = torch.matmul(attn, v)
        else:
            # Use chunked attention
            out = torch.zeros_like(v)
            for i in range(0, N, self.chunk_size):
                end_i = min(i + self.chunk_size, N)
                q_chunk = q[:, :, i:end_i]

                attn_chunk = torch.softmax(
                    torch.matmul(q_chunk, k.transpose(-2, -1)) * scale, dim=-1
                )
                out[:, :, i:end_i] = torch.matmul(attn_chunk, v)

        # Reshape back
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        out = self.proj(out)

        # Residual connection
        return x + out


class ResidualBlock(nn.Module):
    """
    Residual block for enhanced network expressiveness.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class UltraHighCapacityDepthHead(nn.Module):
    """
    High-capacity depth prediction head.

    Input: DINOv2-L features (B, 1024, 19, 23)
    Output: depth map (B, 1, 256, 320)
    """

    def __init__(self, in_channels=1024, target_size=(256, 320)):
        super().__init__()
        self.target_size = target_size

        # Stage 1: High-capacity feature extraction (19x23)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, 2048, 3, 1, 1),
            nn.GroupNorm(64, 2048),
            nn.GELU(),

            nn.Conv2d(2048, 2048, 3, 1, 1),
            nn.GroupNorm(64, 2048),
            nn.GELU(),

            nn.Conv2d(2048, 2048, 3, 1, 1),
            nn.GroupNorm(64, 2048),
            nn.GELU(),

            nn.Conv2d(2048, 1024, 3, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
        )

        # Stage 2: Learnable upsampling + residual enhancement

        # First upsample: 19x23 -> 38x46 (2x)
        self.upsample1 = PixelShuffleUpsample(1024, 512, upscale_factor=2)
        self.residual1 = nn.Sequential(
            ResidualBlock(512, dilation=1),
            ResidualBlock(512, dilation=2),  # Dilated convolution for larger receptive field
            ResidualBlock(512, dilation=1),
        )

        # Second upsample: 38x46 -> 76x92 (2x)
        self.upsample2 = PixelShuffleUpsample(512, 256, upscale_factor=2)
        self.residual2 = nn.Sequential(
            ResidualBlock(256, dilation=1),
            ResidualBlock(256, dilation=2),
            ResidualBlock(256, dilation=4),
            ResidualBlock(256, dilation=1),
        )

        # Third upsample: 76x92 -> 152x184 (2x)
        self.upsample3 = PixelShuffleUpsample(256, 128, upscale_factor=2)
        self.residual3 = nn.Sequential(
            ResidualBlock(128, dilation=1),
            ResidualBlock(128, dilation=2),
            ResidualBlock(128, dilation=1),
        )

        # Stage 3: Efficient attention enhancement
        self.attention = EfficientSelfAttention2D(128, num_heads=8)

        # Stage 4: Final refinement network
        self.final_refinement = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.GroupNorm(16, 256),
            nn.GELU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GroupNorm(16, 256),
            nn.GELU(),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, x):
        """
        Args:
            x: DINOv2 features (B, 1024, 19, 23)

        Returns:
            depth: depth map (B, 1, target_H, target_W)
        """
        # Stage 1: Feature extraction
        x = self.feature_extraction(x)  # (B, 1024, 19, 23)

        # Stage 2: Progressive upsampling + residual enhancement
        x = self.upsample1(x)     # (B, 512, 38, 46)
        x = self.residual1(x)

        x = self.upsample2(x)     # (B, 256, 76, 92)
        x = self.residual2(x)

        x = self.upsample3(x)     # (B, 128, 152, 184)
        x = self.residual3(x)

        # Stage 3: Attention enhancement
        x = self.attention(x)     # (B, 128, 152, 184)

        # Stage 4: Upsample to target size (152x184 -> 256x320)
        x = F.interpolate(
            x,
            size=self.target_size,
            mode='bilinear',
            align_corners=True
        )  # (B, 128, 256, 320)

        # Stage 5: Final refinement
        depth = self.final_refinement(x)  # (B, 1, 256, 320)

        return depth


def test_ultra_depth_head():
    """Test UltraHighCapacityDepthHead"""
    print("Testing Ultra High Capacity Depth Head...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depth_head = UltraHighCapacityDepthHead(
        in_channels=1024,
        target_size=(256, 320)
    ).to(device)

    batch_size = 2
    x = torch.randn(batch_size, 1024, 19, 23).to(device)

    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        depth = depth_head(x)

    print(f"Output shape: {depth.shape}")
    print(f"Output range: [{depth.min().item():.4f}, {depth.max().item():.4f}]")

    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak GPU memory: {memory_mb:.1f} MB")

    print("Test completed successfully!")

    return depth_head


if __name__ == "__main__":
    test_ultra_depth_head()
