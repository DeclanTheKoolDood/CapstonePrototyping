import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any, Union

class GroupedSelfAttention(nn.Module):
    """
    Multi-Query / Grouped Attention implementation
    Uses shared key/value pairs across attention heads to reduce memory usage
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        dropout_rate: float = 0.05
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = dim // num_heads

        # Ensure num_heads is divisible by num_kv_groups
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        # Query projection for all heads
        self.q_proj = nn.Linear(dim, dim)

        # Key and value projections (shared across groups)
        kv_dim = dim * num_kv_groups // num_heads
        self.k_proj = nn.Linear(dim, kv_dim)
        self.v_proj = nn.Linear(dim, kv_dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim)

        # Reshape for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_groups, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_groups, seq_len, head_dim]

        # Expand k and v to match number of heads
        heads_per_group = self.num_heads // self.num_kv_groups
        k = k.repeat_interleave(heads_per_group, dim=1)  # [batch, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(heads_per_group, dim=1)  # [batch, num_heads, seq_len, head_dim]

        # Compute scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    """
    Transformer block with grouped self-attention and feed-forward network
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.05
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GroupedSelfAttention(dim, num_heads, num_kv_groups, dropout_rate)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
