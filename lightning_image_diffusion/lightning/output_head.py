import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputHead(nn.Module):
    """Better output head: upsampling with transposed conv and residual block."""
    def __init__(self, in_dim, out_channels, up_factor=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=up_factor, stride=up_factor)
        self.res = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GELU(),
        )
        self.final = nn.Conv2d(in_dim, out_channels, 1)

    def forward(self, x):
        x = self.up(x)
        x = x + self.res(x)
        x = self.final(x)
        return x
