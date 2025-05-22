
import torch
import torch.nn as nn
from lightning.unet_layers import DoubleConv, Down, Up, OutConv
from lightning.dpm_solver_pp import DPMSolverPP
from lightning.patch_utils import extract_patches, reconstruct_from_patches
from transformers import CLIPTokenizer, CLIPTextModel

class UNetWithCLIP(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_c=64, text_embed_dim=768, clip_model_name='openai/clip-vit-base-patch16'):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.up1 = Up(base_c * 8, base_c * 4)
        self.up2 = Up(base_c * 4, base_c * 2)
        self.up3 = Up(base_c * 2, base_c)
        self.outc = OutConv(base_c, out_channels)
        # CLIP encoder
        if CLIPTextModel is not None:
            self.clip_encoder = CLIPTextModel.from_pretrained(clip_model_name)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        else:
            self.clip_encoder = None
            self.clip_tokenizer = None
        # Project CLIP embedding to feature map
        self.cond_proj = nn.Linear(text_embed_dim, base_c * 8)

    def get_text_features(self, text):
        # text: list of strings
        if self.clip_encoder is None or self.clip_tokenizer is None:
            raise RuntimeError("transformers not installed or CLIP model not available")
        device = next(self.parameters()).device
        tokens = self.clip_tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        outputs = self.clip_encoder(input_ids=tokens)
        # Use the [EOS] pooled output (default for CLIPTextModel)
        return outputs.last_hidden_state[:, 0, :]  # (B, text_embed_dim)

    def forward(self, x, text):
        # x: (B, C, H, W), text: list of strings
        cond = self.get_text_features(text)  # (B, text_embed_dim)
        cond = self.cond_proj(cond)[:, :, None, None]  # (B, base_c*8, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # Add CLIP conditioning at bottleneck
        x4 = x4 + cond
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

class LightningImageDiffusion:
    def __init__(self, model_ckpt=None, device='cuda', clip_model_name='openai/clip-vit-base-patch16'):
        self.device = device
        self.model = UNetWithCLIP(clip_model_name=clip_model_name).to(device)
        if model_ckpt:
            self.model.load_state_dict(torch.load(model_ckpt, map_location=device))
        self.model.eval()
        self.scheduler = DPMSolverPP(num_inference_steps=10)


    @torch.no_grad()
    def sample(self, shape, text, patch_size=32, overlap=8, num_inference_steps=5):
        # Fastest diffusion: use very few steps, DPM-Solver++
        img = torch.randn((1, 3, *shape), device=self.device)
        patches, positions, out_shape = extract_patches(img, patch_size, overlap)
        self.scheduler.num_inference_steps = num_inference_steps
        timesteps = self.scheduler.get_timesteps()
        for t in timesteps:
            for i in range(patches.shape[0]):
                patch = patches[i:i+1]
                model_out = self.model(patch, text)
                patches[i:i+1] = torch.tensor(
                    self.scheduler.step(model_out.cpu().numpy(), t, patch.cpu().numpy()),
                    device=self.device
                )
        img = reconstruct_from_patches(patches, positions, out_shape, patch_size, overlap)
        return img.clamp(-1, 1)

    @torch.no_grad()
    def denoise(self, img, text, patch_size=32, overlap=8):
        # img: (1, 3, H, W) in [-1, 1], text: list of strings
        patches, positions, out_shape = extract_patches(img, patch_size, overlap)
        for t in self.scheduler.get_timesteps():
            for i in range(patches.shape[0]):
                patch = patches[i:i+1]
                model_out = self.model(patch, text)
                patches[i:i+1] = torch.tensor(
                    self.scheduler.step(model_out.cpu().numpy(), t, patch.cpu().numpy()),
                    device=self.device
                )
        img = reconstruct_from_patches(patches, positions, out_shape, patch_size, overlap)
        return img.clamp(-1, 1)
