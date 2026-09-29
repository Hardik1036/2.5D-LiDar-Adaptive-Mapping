import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """conv1-bn1-relu-conv2-bn2 + residual add, matches blocks.N.conv1/bn1/conv2/bn2"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ThreatNet1D(nn.Module):
    """
    Lightweight 1D residual CNN for low-profile threat detection.
    Input:  (B, 2, 512)  -> [ground_residual, normalized_intensity]
    Output: (B, 1, 512)  -> per-point threat logit (apply sigmoid for probability)
    Reconstructed to exactly match best_model_v3.pt's state_dict.
    """
    def __init__(self, in_channels=2, hidden=16, num_blocks=3, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden, kernel_size) for _ in range(num_blocks)
        ])
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
