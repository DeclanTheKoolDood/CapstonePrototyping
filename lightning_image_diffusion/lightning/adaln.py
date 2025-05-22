import torch
import torch.nn as nn

class AdaLayerNorm(nn.Module):
    """Conditional LayerNorm (AdaLN/FiLM style)"""
    def __init__(self, normalized_shape, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        self.gamma = nn.Linear(cond_dim, normalized_shape)
        self.beta = nn.Linear(cond_dim, normalized_shape)

    def forward(self, x, cond):
        # x: (B, N, D), cond: (B, cond_dim)
        x = self.norm(x)
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return x * (1 + gamma) + beta
