
from PIL import Image
from lightning.dit32x32 import HunyuanDiT
from lightning.sdxl_vae import SDXLVAE
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.optimization import Adafactor

import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Data augmentation pipeline
# --- Data Augmentation Pipeline ---
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class AddPoissonNoise(object):
    def __init__(self, lam=10):
        self.lam = lam
    def __call__(self, tensor):
        clamped = torch.clamp(tensor, min=0)
        noisy = torch.poisson(clamped * self.lam) / self.lam
        return noisy

class GridMask(object):
    def __init__(self, d_min=32, d_max=64, ratio=0.6, rotate=1, mode=0, prob=0.5):
        self.d_min = d_min
        self.d_max = d_max
        self.ratio = ratio
        self.rotate = rotate
        self.mode = mode
        self.prob = prob
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        h, w = img.shape[1:]
        d = random.randint(self.d_min, self.d_max)
        l = int(d * self.ratio + 0.5)
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)
        mask = torch.ones((h, w), device=img.device)
        for i in range(-1, h // d + 1):
            s = d * i + st_h
            t = s + l
            s = max(s, 0)
            t = min(t, h)
            mask[s:t, :] = 0
        for i in range(-1, w // d + 1):
            s = d * i + st_w
            t = s + l
            s = max(s, 0)
            t = min(t, w)
            mask[:, s:t] = 0
        mask = mask.expand_as(img)
        return img * mask


# --- Fix: Replace all transforms.Lambda(lambda ...) with picklable functions/classes ---
def random_gamma(x):
    gamma = random.uniform(0.8, 1.2)
    return transforms.functional.adjust_gamma(x, gamma)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    random_gamma,
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    AddGaussianNoise(),
    AddPoissonNoise(),
    GridMask(),
])

# Patch-based MSE loss
class PatchMSELoss(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.mse = nn.MSELoss()
    def forward(self, x, y):
        # x, y: (B, C, H, W)
        B, C, H, W = x.shape
        loss = 0.0
        count = 0
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                x_patch = x[..., i:i+self.patch_size, j:j+self.patch_size]
                y_patch = y[..., i:i+self.patch_size, j:j+self.patch_size]
                if x_patch.shape[-1] == self.patch_size and x_patch.shape[-2] == self.patch_size:
                    loss = loss + self.mse(x_patch, y_patch)
                    count += 1
        return loss / max(count, 1)

# EMA weights
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    def copy_to(self):
        self.model.load_state_dict(self.shadow)

# Profiling context manager
class Profiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.prof = None
    def __enter__(self):
        if self.enabled:
            import torch.profiler
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,
                with_stack=True)
            self.prof.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.prof:
            self.prof.__exit__(exc_type, exc_val, exc_tb)

# Basic training script with all features

def train(
    model, vae, dataloader, epochs=10, lr=1e-4, device='cuda',
    grad_accum_steps=2, use_amp=True, use_profiler=False,
    early_stopping_patience=5, checkpoint_dir='./checkpoints',
    scheduler_type='cosine', warmup_steps=500, max_steps=10000
):
    scaler = GradScaler(enabled=use_amp)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/warmup_steps, 1.0))
    ema = EMA(model)
    loss_fn = PatchMSELoss(patch_size=32)
    best_loss = float('inf')
    patience = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    step = 0
    with Profiler(enabled=use_profiler):
        for epoch in range(epochs):
            for i, (img, texts) in enumerate(dataloader):
                img = img.to(device)
                # Use real captions from dataset
                cond = clip_text_encoder(list(texts), device)
                with torch.no_grad():
                    target_latent = vae.encode(img.float())
                with autocast(device_type=device, enabled=use_amp):
                    # Gradient checkpointing for memory efficiency
                    def custom_forward(*inputs):
                        # Ensure input dtype matches model parameters (important for AMP)
                        model_device = next(model.parameters()).device
                        model_dtype = next(model.parameters()).dtype
                        img_cast = inputs[0].to(device=model_device, dtype=model_dtype)
                        return model(img_cast, inputs[1])
                    output = torch.utils.checkpoint.checkpoint(custom_forward, img, cond)
                    output_latent = vae.encode(output.float())
                    loss = loss_fn(output_latent, target_latent) / grad_accum_steps
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update()
                    scheduler.step()
                    step += 1
                if step % 100 == 0:
                    print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
                # Early stopping and checkpointing
                if step % 500 == 0:
                    val_loss = loss.item()  # Replace with real validation loss
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience = 0
                        torch.save(model.state_dict(), f"{checkpoint_dir}/best.pt")
                    else:
                        patience += 1
                    if patience > early_stopping_patience:
                        print("Early stopping triggered.")
                        return
                if step >= max_steps:
                    print("Max steps reached.")
                    return

class ImageCaptionFolderDataset(Dataset):
    def __init__(self, folder, image_transform=None, caption_transform=None, exts={'.png', '.jpg', '.jpeg', '.bmp'}):
        self.folder = folder
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.exts = exts
        self.samples = []
        for fname in os.listdir(folder):
            base, ext = os.path.splitext(fname)
            if ext.lower() in self.exts:
                img_path = os.path.join(folder, fname)
                txt_path = os.path.join(folder, base + '.txt')
                if os.path.isfile(txt_path):
                    self.samples.append((img_path, txt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        if self.image_transform:
            image = self.image_transform(image)
        if self.caption_transform:
            caption = self.caption_transform(caption)
        return image, caption

# Example usage (replace with your dataset and model)
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Custom dataset for image-caption folder

    # Set your folder path here:
    image_caption_folder = 'dataset'  # <-- Change this to your folder path
    dataset = ImageCaptionFolderDataset(image_caption_folder, image_transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    vae = SDXLVAE(device=device)
    model = HunyuanDiT().to(device)

    # Initialize CLIP text encoder and tokenizer
    clip_model_name = 'openai/clip-vit-base-patch16'
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    clip_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)

    def clip_text_encoder(texts, device):
        tokens = clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        outputs = clip_encoder(input_ids=tokens)
        return outputs.last_hidden_state[:, 0, :]  # (B, 768)

    train(model, vae, dataloader, device=device)
