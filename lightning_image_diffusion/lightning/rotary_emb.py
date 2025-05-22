import torch
import torch.nn as nn
import torch.nn.functional as F

# Rotary positional embedding (2D)
def apply_rotary_emb(x, sin, cos):
    # x: (B, N, D), sin/cos: (N, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    # Returns (grid_size*grid_size, embed_dim)
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 1])
    return torch.cat([emb_h, emb_w], dim=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # pos: (M,)
    omega = torch.arange(embed_dim) / embed_dim
    omega = 1. / (10000 ** omega)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)
