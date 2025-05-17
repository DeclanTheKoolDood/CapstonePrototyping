import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

# Set up logging
logger = logging.getLogger(__name__)

class AdaptiveLayerNorm(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) implementation
    Applies affine transformation to input features based on conditioning
    """
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.scale_shift = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization

        Args:
            x: Input tensor of shape [batch_size*height*width, channels] or [batch_size, height, width, channels]
            condition: Conditioning tensor of shape [batch_size, condition_dim]

        Returns:
            Normalized and modulated tensor of the same shape as x
        """
        # Store original dtype for output consistency
        original_dtype = x.dtype

        # Apply layer normalization
        normalized = self.norm(x)

        # Add debug logging
        logger.debug(f"AdaptiveLayerNorm input shapes: x={x.shape}, condition={condition.shape}")

        # For spatial inputs (B*H*W, C), we need to repeat the condition for each spatial position
        if len(x.shape) == 2 and len(condition.shape) == 2:
            # Get batch size from condition and calculate spatial size
            batch_size = condition.shape[0]

            # Check if x can be evenly divided by batch_size
            if x.shape[0] % batch_size != 0:
                logger.warning(f"Input shape {x.shape[0]} not divisible by batch size {batch_size}. Using batch_size=1")
                batch_size = 1

            spatial_size = x.shape[0] // batch_size
            logger.debug(f"Calculated spatial_size={spatial_size} from x.shape[0]={x.shape[0]} and batch_size={batch_size}")

            try:
                # Repeat condition for each spatial position
                condition_expanded = condition.repeat_interleave(spatial_size, dim=0)
                logger.debug(f"Expanded condition shape: {condition_expanded.shape}")

                # Generate scale and shift parameters from expanded condition
                scale_shift = self.scale_shift(condition_expanded)
            except Exception as e:
                logger.error(f"Error in AdaptiveLayerNorm: {e}")
                logger.error(f"x.shape={x.shape}, condition.shape={condition.shape}, batch_size={batch_size}, spatial_size={spatial_size}")
                # Fallback to using the first condition for all spatial positions
                condition_first = condition[0:1].expand(x.shape[0], -1)
                scale_shift = self.scale_shift(condition_first)
        else:
            # Generate scale and shift parameters from condition as is
            logger.debug(f"Using condition as is, shape: {condition.shape}")
            scale_shift = self.scale_shift(condition)

        # Split into scale and shift
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)

        # Apply FiLM transformation
        result = normalized * (1 + scale) + shift

        # Return result in the original dtype for consistency
        return result.to(dtype=original_dtype)

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block with multiple internal connections
    Improves gradient flow and feature reuse
    """
    def __init__(self, channels: int, growth_rate: int, num_layers: int = 4, dropout_rate: float = 0.05):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout_rate)
            ))
        self.local_fusion = nn.Conv2d(channels + num_layers * growth_rate, channels, kernel_size=1)
        self.global_residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original dtype for output consistency
        original_dtype = x.dtype

        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        fused_features = self.local_fusion(torch.cat(features, dim=1))
        result = fused_features + self.global_residual(x)

        # Return result in the original dtype for consistency
        return result.to(dtype=original_dtype)

class ScalePredictor(nn.Module):
    """
    Predicts the next scale from the current scale
    Uses residual connections for upscaling
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        num_blocks: int = 3,
        dropout_rate: float = 0.05
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Residual dense blocks
        self.res_blocks = nn.ModuleList([
            ResidualDenseBlock(64, 32, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        # Upsampling layer
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor * scale_factor, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.GELU()
        )

        # Residual prediction head
        self.res_pred = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Store original dtype for output consistency
        original_dtype = x.dtype

        # Initial feature extraction
        features = self.initial_conv(x)

        # Apply residual dense blocks
        for block in self.res_blocks:
            features = block(features)

        # Upsample features
        upsampled_features = self.upsample(features)

        # Generate residual prediction
        residual = self.res_pred(upsampled_features)

        # Upsample input using bilinear interpolation for residual connection
        upsampled_input = F.interpolate(
            x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )

        # If channel dimensions don't match, pad or project the upsampled input
        if upsampled_input.shape[1] != residual.shape[1]:
            if upsampled_input.shape[1] < residual.shape[1]:
                # Pad with zeros if input has fewer channels
                padding = torch.zeros(
                    upsampled_input.shape[0],
                    residual.shape[1] - upsampled_input.shape[1],
                    upsampled_input.shape[2],
                    upsampled_input.shape[3],
                    device=upsampled_input.device,
                    dtype=upsampled_input.dtype
                )
                upsampled_input = torch.cat([upsampled_input, padding], dim=1)
            else:
                # Project to fewer channels if input has more channels
                upsampled_input = upsampled_input[:, :residual.shape[1], :, :]

        # Combine upsampled input with residual prediction
        output = upsampled_input + residual

        # Return output in the original dtype for consistency
        return output.to(dtype=original_dtype), residual.to(dtype=original_dtype)
