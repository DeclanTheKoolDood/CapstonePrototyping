import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScalePatchEmbed(nn.Module):
    """Hierarchical patch embedding: stack multiple patch embeddings for multi-scale features."""
    def __init__(self, in_channels, embed_dim, patch_sizes=(4, 8)):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=ps, stride=ps) for ps in patch_sizes
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        outs = []
        for embed in self.embeds:
            out = embed(x)
            B, C, H, W = out.shape
            out = out.flatten(2).transpose(1, 2)  # (B, N, C)
            outs.append(out)
        x = torch.cat(outs, dim=1)  # (B, sum(N), C)
        x = self.norm(x)
        return x
