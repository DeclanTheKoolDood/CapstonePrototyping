

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.rotary_emb import apply_rotary_emb, get_2d_sincos_pos_embed
from lightning.adaln import AdaLayerNorm
from lightning.multi_scale_patch import MultiScalePatchEmbed
from lightning.output_head import OutputHead
from lightning.sdxl_vae import SDXLVAE
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

class FlashAttention(nn.Module):
    """
    FlashAttention using the flash-attn library if available, else fallback to standard attention.
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        H = self.heads
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.view(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D_head)
        if FLASH_ATTN_AVAILABLE and x.is_cuda:
            # flash_attn_func expects (B*H, N, D_head)
            q = q.reshape(B * H, N, self.head_dim)
            k = k.reshape(B * H, N, self.head_dim)
            v = v.reshape(B * H, N, self.head_dim)
            out = flash_attn_func(q, k, v, causal=False)
            out = out.reshape(B, H, N, self.head_dim).transpose(1, 2).reshape(B, N, D)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
            attn = torch.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn, v)  # (B, H, N, D_head)
            out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out


# Parameter sharing block for DiT (shared weights for all blocks)

# DiT block with rotary embedding and AdaLayerNorm (conditional)
class SharedDiTBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0, cond_dim=768, layer_scale_init=1e-6):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = FlashAttention(dim, heads)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)

    def forward(self, x, cond, rotary_sin=None, rotary_cos=None):
        h = x
        x = self.norm1(x, cond)
        if rotary_sin is not None and rotary_cos is not None:
            x = apply_rotary_emb(x, rotary_sin, rotary_cos)
        x = self.attn(x)
        x = h + self.gamma_1 * x
        h = x
        x = self.norm2(x, cond)
        x = self.mlp(x)
        x = h + self.gamma_2 * x
        return x


# Hunyuan DiT with flash-attn, efficient patch embedding, layer scaling, dynamic pos emb, parameter sharing, efficient output head, SDXL1.0 value
class EfficientPatchEmbed(nn.Module):
    """Efficient patch embedding using depthwise separable conv."""
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class HunyuanDiT(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, dim=256, depth=8, heads=4, mlp_ratio=4.0, cond_dim=512, use_rotary=True, sdxl_value=0.18215):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.sdxl_value = sdxl_value
        self.vae = SDXLVAE()
        # SDXL VAE outputs 4-channel latents
        self.patch_embed = MultiScalePatchEmbed(4, dim, patch_sizes=(4, 8))
        self.use_rotary = use_rotary
        self.depth = depth
        self.shared_block = SharedDiTBlock(dim, heads, mlp_ratio, cond_dim)
        self.norm = nn.LayerNorm(dim)
        self.output_head = OutputHead(dim, in_channels * patch_size * patch_size, up_factor=patch_size)

    def forward(self, x, cond):
        # x: (B, 3, H, W), cond: (B, cond_dim)
        # Step 1: Encode image to latents using VAE
        z = self.vae.encode(x)  # (B, latent_dim, H//8, W//8)
        # Ensure z matches model dtype/device for patch embedding
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        z = z.to(device=model_device, dtype=model_dtype)
        # Step 2: Patchify latents
        B, C, H, W = z.shape
        x = self.patch_embed(z)  # (B, N, dim)
        N = x.shape[1]
        # Step 3: Rotary positional embedding
        rotary_sin, rotary_cos = None, None
        if self.use_rotary:
            grid_size = int(N ** 0.5)
            pos_emb = get_2d_sincos_pos_embed(self.dim, grid_size).to(x.device)
            sin, cos = pos_emb.sin(), pos_emb.cos()
            rotary_sin, rotary_cos = sin, cos
        # Step 4: Transformer blocks with AdaLN
        for _ in range(self.depth):
            x = self.shared_block(x, cond, rotary_sin, rotary_cos)
        x = self.norm(x)
        # Step 5: Unpatchify
        Hp = Wp = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, self.dim, Hp, Wp)
        # Step 6: Output head (upsample to VAE latent size)
        x = self.output_head(x)
        # Step 7: Decode latents to image
        x = self.vae.decode(x)
        # Step 8: SDXL1.0 value scaling
        x = x * self.sdxl_value
        return x
