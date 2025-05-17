import torch
import torch.nn as nn
import math
import logging
from typing import List, Tuple, Optional, Union

from .layers import AdaptiveLayerNorm, ResidualDenseBlock, ScalePredictor
from .encoders import CLIPTextEncoderWrapper, SDXLVAEWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NextScalePredictionModel(nn.Module):
    """
    Main model for Next Scale Prediction
    Progressively predicts larger scale images from smaller ones
    """
    def __init__(
        self,
        initial_size: int = 16,
        target_size: int = 1024,
        scale_factor: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        condition_dim: int = 512,
        dropout_rate: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ):
        super().__init__()
        self.initial_size = initial_size
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        # Always use float32 internally to avoid dtype mismatches
        self.dtype = torch.float32

        # Calculate number of upscaling steps needed
        num_steps = int(math.log(target_size // initial_size, scale_factor))

        # Text encoder for conditioning
        self.text_encoder = CLIPTextEncoderWrapper(
            model_name=clip_model_name,
            device=device,
            dtype=dtype
        )

        # Condition projection for FiLM layers
        self.condition_proj = nn.Linear(self.text_encoder.embed_dim, condition_dim)

        # Initial noise to image projection
        self.initial_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

        # Create scale predictors for each resolution
        self.scale_predictors = nn.ModuleList()
        current_size = initial_size

        for i in range(num_steps):
            # Determine if we need to change channel dimensions at certain scales
            # For example, we might want more channels at lower resolutions
            if current_size < 64:
                pred_in_channels = out_channels
                pred_out_channels = out_channels
            else:
                pred_in_channels = out_channels
                pred_out_channels = out_channels

            # Create scale predictor
            self.scale_predictors.append(
                ScalePredictor(
                    in_channels=pred_in_channels,
                    out_channels=pred_out_channels,
                    scale_factor=scale_factor,
                    dropout_rate=dropout_rate
                )
            )

            # Update current size for next iteration
            current_size *= scale_factor

        # Adaptive layer norms for text conditioning at each scale
        self.adaptive_norms = nn.ModuleList([
            AdaptiveLayerNorm(out_channels, condition_dim)
            for _ in range(num_steps + 1)  # +1 for initial projection
        ])

        # Final refinement for output
        self.final_refinement = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualDenseBlock(64, 32, dropout_rate=dropout_rate),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

        # Move model to device first
        self.to(device=device)

        # Force all parameters to float32 regardless of the requested dtype
        # This is to avoid the "Input type (float) and bias type (struct c10::Half)" error
        self._convert_all_parameters_to_float32()

    def _convert_all_parameters_to_float32(self):
        """
        Recursively convert all parameters in the model to float32.
        This is a more thorough approach than just converting weights and biases.
        """
        # First, convert all parameters in the model
        for param in self.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(dtype=torch.float32)

        # Then, explicitly handle Sequential modules and their children
        for name, module in self.named_modules():
            if isinstance(module, nn.Sequential):
                for child in module.children():
                    if hasattr(child, 'weight') and child.weight is not None:
                        child.weight.data = child.weight.data.to(dtype=torch.float32)
                    if hasattr(child, 'bias') and child.bias is not None:
                        child.bias.data = child.bias.data.to(dtype=torch.float32)

            # Handle specific module types directly
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.to(dtype=torch.float32)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.to(dtype=torch.float32)

        # Explicitly handle the initial_proj module
        for layer in self.initial_proj:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data = layer.weight.data.to(dtype=torch.float32)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype=torch.float32)

        # Explicitly handle the final_refinement module
        for layer in self.final_refinement:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data = layer.weight.data.to(dtype=torch.float32)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype=torch.float32)

        # Log completion
        logger.info("Converted all model parameters to float32")

    def get_condition(self, text_prompts: Optional[List[str]] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Get conditioning vector from text prompts

        Args:
            text_prompts: Optional list of text prompts
            batch_size: Batch size (used if text_prompts is None)

        Returns:
            Conditioning vector of shape [batch_size, condition_dim]
        """
        if text_prompts is not None:
            # Get text embeddings from encoder
            text_embed = self.text_encoder(text_prompts)
            # Ensure text embeddings are float32 for consistent computation
            text_embed = text_embed.to(dtype=torch.float32)
            # Project to condition dimension
            condition = self.condition_proj(text_embed)
        else:
            # Create dummy condition if no text is provided
            # Always use float32 for consistency
            condition = torch.zeros(batch_size, 512, device=self.device, dtype=torch.float32)

        return condition

    def forward(
        self,
        x: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Store original dtype for output consistency
        original_dtype = x.dtype

        # Move input to device and ensure it's float32 to avoid dtype mismatches
        x = x.to(device=self.device, dtype=torch.float32)

        batch_size = x.shape[0]

        # Get conditioning vector
        condition = self.get_condition(text_prompts, batch_size)

        # Initial projection from noise to starting image
        current = self.initial_proj(x)

        # Apply adaptive norm with conditioning

        # Reshape image tensor for layer norm
        current_flat = current.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Apply adaptive layer norm to each spatial position
        batch_size, h, w, channels = current_flat.shape
        current_flat_reshaped = current_flat.reshape(-1, channels)  # [B*H*W, C]

        # Apply adaptive layer norm
        current_flat_reshaped = self.adaptive_norms[0](current_flat_reshaped, condition)

        # Reshape back to original shape
        current_flat = current_flat_reshaped.reshape(batch_size, h, w, channels)

        # Permute back to [B, C, H, W]
        current = current_flat.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Store intermediate outputs if requested
        intermediates = [current.to(dtype=original_dtype)] if return_intermediates else None

        # Apply scale predictors sequentially
        for i, predictor in enumerate(self.scale_predictors):
            # Predict next scale
            current, _ = predictor(current)

            # Apply adaptive norm with conditioning

            # Reshape image tensor for layer norm
            current_flat = current.permute(0, 2, 3, 1)  # [B, H, W, C]

            # Apply adaptive layer norm to each spatial position
            batch_size, h, w, channels = current_flat.shape
            current_flat_reshaped = current_flat.reshape(-1, channels)  # [B*H*W, C]

            # Apply adaptive layer norm
            current_flat_reshaped = self.adaptive_norms[i+1](current_flat_reshaped, condition)

            # Reshape back to original shape
            current_flat = current_flat_reshaped.reshape(batch_size, h, w, channels)

            # Permute back to [B, C, H, W]
            current = current_flat.permute(0, 3, 1, 2)  # [B, C, H, W]

            if return_intermediates:
                intermediates.append(current.to(dtype=original_dtype))

        # Apply final refinement
        output = self.final_refinement(current)

        # Return output in the original dtype for consistency
        output = output.to(dtype=original_dtype)

        if return_intermediates:
            return output, intermediates
        return output

# Generation functions
def generate_image(
    model: NextScalePredictionModel,
    vae: SDXLVAEWrapper,
    batch_size: int = 1,
    initial_size: int = 16,
    text_prompts: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: Optional[int] = None,
    dtype: torch.dtype = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate images using the Next Scale Prediction model

    Args:
        model: The trained NextScalePredictionModel
        vae: SDXL VAE model for final decoding
        batch_size: Number of images to generate
        initial_size: Size of initial noise latent
        text_prompts: Optional list of text prompts for conditioning (one per batch item)
        device: Device to run generation on
        seed: Optional random seed for reproducibility
        dtype: Optional dtype for generation (defaults to model's dtype)

    Returns:
        Tuple containing:
        - Generated images of shape [batch_size, 3, target_size, target_size]
        - List of intermediate outputs at each scale
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Use model's dtype if not specified
    if dtype is None:
        dtype = model.dtype

    # Create random noise as starting point
    noise = torch.randn(batch_size, 3, initial_size, initial_size, device=device, dtype=dtype)

    # Prepare text prompts if provided
    if text_prompts is None and batch_size > 1:
        # If batch_size > 1 but no prompts provided, use empty strings
        text_prompts = [""] * batch_size
    elif text_prompts is not None and len(text_prompts) == 1 and batch_size > 1:
        # If only one prompt provided but batch_size > 1, repeat the prompt
        text_prompts = text_prompts * batch_size

    # Generate image using the model
    with torch.no_grad():
        logger.info(f"Generating {batch_size} images with initial size {initial_size}x{initial_size}")
        latents, intermediates = model(noise, text_prompts, return_intermediates=True)

        # Decode final latents using VAE
        logger.info("Decoding latents with SDXL VAE")
        images = vae.decode(latents)

    return images, intermediates

def create_model_and_vae(
    initial_size: int = 16,
    target_size: int = 1024,
    scale_factor: int = 2,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    vae_model_name: str = "stabilityai/sdxl-vae",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32  # Always use float32 to avoid dtype mismatches
) -> Tuple[NextScalePredictionModel, SDXLVAEWrapper]:
    """
    Create and initialize the Next Scale Prediction model and SDXL VAE

    Args:
        initial_size: Size of initial noise latent
        target_size: Target image size
        scale_factor: Factor by which to scale at each step
        clip_model_name: Name of CLIP model to use for text encoding
        vae_model_name: Name of VAE model to use for decoding
        device: Device to run model on
        dtype: Data type to use for model parameters

    Returns:
        Tuple containing:
        - Initialized NextScalePredictionModel
        - Initialized SDXLVAEWrapper
    """
    logger.info(f"Creating model with initial_size={initial_size}, target_size={target_size}")

    # Create model
    model = NextScalePredictionModel(
        initial_size=initial_size,
        target_size=target_size,
        scale_factor=scale_factor,
        clip_model_name=clip_model_name,
        device=device,
        dtype=dtype
    )

    # Create VAE
    vae = SDXLVAEWrapper(
        pretrained_path=vae_model_name,
        device=device,
        dtype=dtype
    )

    return model, vae