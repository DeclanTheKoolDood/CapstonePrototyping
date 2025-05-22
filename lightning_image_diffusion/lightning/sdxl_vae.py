from diffusers import AutoencoderKL
import torch

class SDXLVAE:
    """Wrapper for SDXL1.0 VAE using diffusers' AutoencoderKL."""
    def __init__(self, pretrained_model_name_or_path="stabilityai/sdxl-vae", torch_dtype=torch.float32, device=None):
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype)
        if device is not None:
            self.vae = self.vae.to(device)
        self.latent_scaling = 0.18215

    def encode(self, x):
        # x: (B, 3, H, W), returns (B, latent_dim, H//8, W//8)
        with torch.no_grad():
            vae_device = next(self.vae.parameters()).device
            vae_dtype = next(self.vae.parameters()).dtype
            x = x.to(device=vae_device, dtype=vae_dtype)
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample() * self.latent_scaling
        return z

    def decode(self, z):
        # z: (B, latent_dim, H//8, W//8)
        with torch.no_grad():
            x = self.vae.decode(z / self.latent_scaling).sample
        return x
