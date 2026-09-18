"""
Lightweight SalsaNext Architecture for LiDAR Semantic Segmentation.
Supports both PyTorch nn.Module instantiation and weight loading for DRDO SIH 2026 PS 53.
"""

from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


if HAS_TORCH and nn is not None:
    class SalsaNextBlock(nn.Module):
        """Standard dilated convolution residual block for SalsaNext."""
        def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.2),
            )
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x) + self.shortcut(x)

    class LightweightSalsaNext(nn.Module):
        """
        Lightweight SalsaNext-style LiDAR range image semantic segmentation network.
        Input: (B, in_channels, H, W) e.g. (B, 5, 64, 1024) [range, x, y, z, intensity]
        Output: (B, num_classes, H, W) e.g. (B, 4, 64, 1024) logits
        """
        def __init__(self, in_channels: int = 5, num_classes: int = 4, base_channels: int = 32):
            super().__init__()
            self.in_channels = in_channels
            self.num_classes = num_classes

            # Encoder stages
            self.enc1 = SalsaNextBlock(in_channels, base_channels, dilation=1)
            self.enc2 = SalsaNextBlock(base_channels, base_channels * 2, dilation=2)
            self.enc3 = SalsaNextBlock(base_channels * 2, base_channels * 4, dilation=2)

            # Bottleneck
            self.bottleneck = SalsaNextBlock(base_channels * 4, base_channels * 4, dilation=4)

            # Decoder stages
            self.dec3 = SalsaNextBlock(base_channels * 4, base_channels * 2, dilation=2)
            self.dec2 = SalsaNextBlock(base_channels * 2, base_channels, dilation=1)
            self.dec1 = SalsaNextBlock(base_channels, base_channels, dilation=1)

            # Head / Classifier
            self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            b = self.bottleneck(e3)
            d3 = self.dec3(b + e3)
            d2 = self.dec2(d3 + e2)
            d1 = self.dec1(d2 + e1)
            out = self.classifier(d1)
            return out

    SalsaNextArchitecture = LightweightSalsaNext

else:
    # Stub classes when torch is not available
    class LightweightSalsaNext:
        def __init__(self, *args, **kwargs):
            pass

    SalsaNextArchitecture = LightweightSalsaNext

__all__ = ["LightweightSalsaNext", "SalsaNextArchitecture"]
