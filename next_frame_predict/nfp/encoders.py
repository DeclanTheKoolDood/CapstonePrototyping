import torch
import torch.nn as nn
import logging
from typing import List
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPTextEncoderWrapper(nn.Module):
    """
    Wrapper for CLIP text encoder from Hugging Face Transformers
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading CLIP tokenizer and text encoder from {model_name}")
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_name,
                torch_dtype=dtype
            ).to(device)

            # Freeze the text encoder parameters
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            logger.info("CLIP text encoder loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP text encoder: {e}")
            raise

        # Get embedding dimension from the model
        self.embed_dim = self.text_encoder.config.hidden_size

    def forward(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts using CLIP text encoder

        Args:
            text_prompts: List of text prompts to encode

        Returns:
            Text embeddings of shape [batch_size, embed_dim]
        """
        # Tokenize text prompts
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Encode text prompts
        with torch.no_grad():
            text_embeddings = self.text_encoder(**text_inputs).last_hidden_state

        # Use pooled output (CLS token) or mean pooling
        # Here we use mean pooling for consistency with the original implementation
        attention_mask = text_inputs.attention_mask
        text_embeddings = (text_embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return text_embeddings

class SDXLVAEWrapper(nn.Module):
    """
    Wrapper for SDXL1.0 VAE to encode/decode images
    Uses the AutoencoderKL from diffusers
    """
    def __init__(
        self,
        pretrained_path: str = "stabilityai/sdxl-vae",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading SDXL VAE from {pretrained_path}")
        try:
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_path,
                torch_dtype=dtype
            ).to(device)

            # Use the more efficient attention processor
            if hasattr(self.vae, "set_attn_processor"):
                self.vae.set_attn_processor(AttnProcessor2_0())

            # Freeze the VAE parameters
            for param in self.vae.parameters():
                param.requires_grad = False

            logger.info("SDXL VAE loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SDXL VAE: {e}")
            raise

        # Get latent dimensions
        self.latent_channels = self.vae.config.latent_channels
        self.scaling_factor = self.vae.config.scaling_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space

        Args:
            x: Images of shape [batch_size, 3, height, width] in range [0, 1]

        Returns:
            Latents of shape [batch_size, latent_channels, height/8, width/8]
        """
        # Store original dtype for output consistency
        original_dtype = x.dtype

        # Move input to device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Scale input from [0, 1] to [-1, 1]
        x = 2.0 * x - 1.0

        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Return latents in the original dtype for consistency
        return latents.to(dtype=original_dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images

        Args:
            z: Latents of shape [batch_size, latent_channels, height, width]

        Returns:
            Images of shape [batch_size, 3, height*8, width*8] in range [0, 1]
        """
        # Store original dtype for output consistency
        original_dtype = z.dtype

        # Move input to device and dtype
        z = z.to(device=self.device, dtype=self.dtype)

        # Scale latents
        z = z / self.vae.config.scaling_factor

        # Decode latents to images
        with torch.no_grad():
            images = self.vae.decode(z).sample

        # Scale images from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

        # Return images in the original dtype for consistency
        return images.to(dtype=original_dtype)
