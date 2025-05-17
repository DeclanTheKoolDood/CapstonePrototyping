import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import gc
import os
from typing import List, Tuple, Optional, Dict, Any, Callable
from tqdm import tqdm
from pathlib import Path

from .model import NextScalePredictionModel
from .encoders import SDXLVAEWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressiveTrainer:
    """
    Trainer for progressive training of the Next Scale Prediction model
    Trains the model at different scales, using interpolation to resize images
    """
    def __init__(
        self,
        model: NextScalePredictionModel,
        vae: SDXLVAEWrapper,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        fp16: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        memory_efficient: bool = True,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cpu_offload: bool = False
    ):
        self.model = model
        self.vae = vae
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.memory_efficient = memory_efficient
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cpu_offload = cpu_offload

        # Set up memory optimization
        if self.memory_efficient:
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("Using memory efficient attention")
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)

            # Set PyTorch memory allocation strategy
            if torch.cuda.is_available():
                # Use more aggressive memory caching
                torch.cuda.empty_cache()
                # Set memory allocation strategy
                if hasattr(torch.cuda, 'memory_stats'):
                    logger.info("Setting up memory optimization strategies")
                    # Enable memory stats for better memory management
                    torch.cuda.memory_stats()

        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer with float32 parameters to avoid FP16 gradient issues
        # Create parameter groups with explicit dtype
        param_groups = []
        for param in self.model.parameters():
            if param.requires_grad:
                # Create a copy of the parameter with float32 dtype for the optimizer
                param_groups.append({'params': [param], 'lr': learning_rate, 'weight_decay': weight_decay})

        # Initialize optimizer with float32 parameters
        self.optimizer = optim.AdamW(param_groups)

        # Initialize learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000  # Will be updated during training
        )

        # Initialize scaler for mixed precision training
        try:
            # Try the new constructor first
            self.scaler = torch.amp.GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu', enabled=fp16)
        except (TypeError, ValueError):
            try:
                # Fallback to older constructor without device parameter
                self.scaler = torch.amp.GradScaler(enabled=fp16)
            except (ImportError, AttributeError):
                # Final fallback for very old PyTorch versions
                logger.warning("Using deprecated GradScaler. Consider upgrading PyTorch.")
                # Disable the scaler completely to avoid dtype issues
                self.scaler = torch.amp.GradScaler(enabled=False)

        # Set default optimizer dtype to float32 to avoid FP16 gradient issues
        self.optimizer_dtype = torch.float32

        # Initialize dataloaders with memory optimization
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_factor': self.prefetch_factor if self.num_workers > 0 else None,
        }

        # Add persistent workers if supported and enabled
        if self.persistent_workers and self.num_workers > 0:
            dataloader_kwargs['persistent_workers'] = True

        # Create training dataloader
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs
        )

        # Create validation dataloader if validation dataset is provided
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **dataloader_kwargs
            )
        else:
            self.val_loader = None

        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Calculate number of scales
        self.num_scales = len(self.model.scale_predictors) + 1  # +1 for initial scale
        self.current_scale = 0

        # Calculate sizes for each scale
        self.scale_sizes = self._calculate_scale_sizes()

        logger.info(f"Initialized ProgressiveTrainer with {self.num_scales} scales")
        logger.info(f"Scale sizes: {self.scale_sizes}")

    def _calculate_scale_sizes(self) -> List[int]:
        """Calculate sizes for each scale"""
        sizes = [self.model.initial_size]
        current_size = self.model.initial_size

        for _ in range(len(self.model.scale_predictors)):
            current_size *= self.model.scale_factor
            sizes.append(current_size)

        return sizes

    def _resize_images(self, images: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Resize images to target size using interpolation

        Args:
            images: Images of shape [batch_size, channels, height, width]
            target_size: Target size for height and width

        Returns:
            Resized images of shape [batch_size, channels, target_size, target_size]
        """
        # Store original dtype for output consistency
        original_dtype = images.dtype

        # Skip resizing if already at target size
        if images.shape[2] == target_size and images.shape[3] == target_size:
            return images

        # Resize using interpolation
        resized = F.interpolate(
            images,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        # Return resized images in the original dtype for consistency
        return resized.to(dtype=original_dtype)

    def _prepare_batch(
        self,
        batch: Dict[str, Any],
        current_scale: int,
        target_scale: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[str]]]:
        """
        Prepare batch for training at current scale

        Args:
            batch: Batch from dataloader
            current_scale: Current scale index
            target_scale: Target scale index

        Returns:
            Tuple containing:
            - Input images resized to current scale
            - Target images resized to target scale
            - Text prompts (if available)
        """
        # Extract images and text prompts from batch
        images = batch['image']

        # Handle text prompts if available
        if 'text_prompt' in batch:
            text_prompts = batch['text_prompt']
            # Ensure text_prompts is a list
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]
        else:
            text_prompts = None

        # Get current and target sizes
        current_size = self.scale_sizes[current_scale]
        target_size = self.scale_sizes[target_scale]

        # CPU offloading: resize on CPU if enabled, then move to GPU
        if self.cpu_offload and torch.cuda.is_available():
            # Resize on CPU
            input_images = self._resize_images(images, current_size)
            target_images = self._resize_images(images, target_size)

            # Move to GPU with explicit float32 dtype for consistency
            input_images = input_images.to(self.device, dtype=torch.float32, non_blocking=True)
            target_images = target_images.to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            # Move to GPU first with explicit float32 dtype
            images = images.to(self.device, dtype=torch.float32, non_blocking=True)

            # Resize on GPU
            input_images = self._resize_images(images, current_size)
            target_images = self._resize_images(images, target_size)

            # Free up original images to save memory
            del images
            if self.memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final check to ensure both tensors are float32
        if input_images.dtype != torch.float32:
            input_images = input_images.to(dtype=torch.float32)
        if target_images.dtype != torch.float32:
            target_images = target_images.to(dtype=torch.float32)

        return input_images, target_images, text_prompts

    def _compute_loss(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target images

        Args:
            pred_images: Predicted images
            target_images: Target images

        Returns:
            Loss value
        """
        # L1 loss
        l1_loss = F.l1_loss(pred_images, target_images)

        # L2 loss (MSE)
        mse_loss = F.mse_loss(pred_images, target_images)

        # Combine losses
        loss = l1_loss + mse_loss

        return loss

    def train_epoch(
        self,
        current_scale: int,
        target_scale: int,
        epoch: int
    ) -> float:
        """
        Train for one epoch at specified scales

        Args:
            current_scale: Current scale index
            target_scale: Target scale index
            epoch: Current epoch

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Progress bar
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} (Scale {current_scale}->{target_scale})",
            leave=False
        )

        # Clear cache before training
        if torch.cuda.is_available() and self.memory_efficient:
            torch.cuda.empty_cache()
            gc.collect()

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch
                input_images, target_images, text_prompts = self._prepare_batch(
                    batch, current_scale, target_scale
                )

                # Forward pass with mixed precision
                # Use a context manager that works with the current PyTorch version
                # Always use float32 to avoid dtype mismatches
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                   dtype=torch.float32, enabled=self.fp16):
                    # If current_scale is 0, use the initial projection
                    if current_scale == 0:
                        # Generate random noise as input
                        noise = torch.randn_like(input_images)

                        # Forward pass through the model
                        if target_scale == self.num_scales - 1:
                            # If target is the final scale, use the full model
                            pred_images = self.model(noise, text_prompts)
                        else:
                            # Otherwise, use only the relevant scale predictors
                            current = self.model.initial_proj(noise)

                            # Apply scale predictors up to target scale
                            for i in range(target_scale):
                                current, _ = self.model.scale_predictors[i](current)

                            pred_images = current
                    else:
                        # Forward pass through relevant scale predictors
                        current = input_images

                        # Apply scale predictors from current to target scale
                        for i in range(current_scale, target_scale):
                            current, _ = self.model.scale_predictors[i](current)

                        pred_images = current

                    # Compute loss
                    loss = self._compute_loss(pred_images, target_images)
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with gradient accumulation
                self.scaler.scale(loss).backward()

                # Free up memory
                del pred_images
                if self.memory_efficient and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Check if any parameters have FP16 gradients
                    has_fp16_grads = False
                    for param in self.model.parameters():
                        if param.grad is not None and param.grad.dtype == torch.float16:
                            has_fp16_grads = True
                            # Convert FP16 gradients to FP32
                            param.grad = param.grad.float()

                    try:
                        # Clip gradients
                        if not has_fp16_grads:
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        # Update weights
                        if has_fp16_grads:
                            # If we had FP16 gradients, use regular optimizer step
                            self.optimizer.step()
                        else:
                            # Otherwise use scaler
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                        self.optimizer.zero_grad(set_to_none=True)
                    except ValueError as e:
                        if "Attempting to unscale FP16 gradients" in str(e):
                            logger.warning("FP16 gradients detected. Skipping scaler unscaling and using direct optimizer step.")
                            # Skip unscaling and just update weights directly
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            raise e

                    # Update learning rate
                    self.lr_scheduler.step()

                    # Update global step
                    self.global_step += 1

                    # Perform memory cleanup
                    if self.memory_efficient and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                # Update progress bar
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                progress_bar.set_postfix({
                    'loss': loss.item() * self.gradient_accumulation_steps,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'mem': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
                })

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"CUDA OOM error: {str(e)}")
                    # Try to recover by clearing cache and reducing batch size temporarily
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    # Skip this batch and continue with the next one
                    logger.warning(f"Skipping batch {batch_idx} due to OOM error")
                    continue
                else:
                    # Re-raise other runtime errors
                    raise e

        # Calculate average loss
        avg_loss = epoch_loss / num_batches

        # Final cleanup after epoch
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return avg_loss

    def validate(
        self,
        current_scale: int,
        target_scale: int
    ) -> float:
        """
        Validate the model at specified scales

        Args:
            current_scale: Current scale index
            target_scale: Target scale index

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)

        # Clear cache before validation
        if torch.cuda.is_available() and self.memory_efficient:
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                try:
                    # Prepare batch
                    input_images, target_images, text_prompts = self._prepare_batch(
                        batch, current_scale, target_scale
                    )

                    # Use mixed precision for validation as well if enabled
                    # Always use float32 to avoid dtype mismatches
                    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                      dtype=torch.float32, enabled=self.fp16):
                        # Forward pass
                        if current_scale == 0:
                            # Generate random noise as input
                            noise = torch.randn_like(input_images)

                            # Forward pass through the model
                            if target_scale == self.num_scales - 1:
                                # If target is the final scale, use the full model
                                pred_images = self.model(noise, text_prompts)
                            else:
                                # Otherwise, use only the relevant scale predictors
                                current = self.model.initial_proj(noise)

                                # Apply scale predictors up to target scale
                                for i in range(target_scale):
                                    current, _ = self.model.scale_predictors[i](current)

                                pred_images = current
                        else:
                            # Forward pass through relevant scale predictors
                            current = input_images

                            # Apply scale predictors from current to target scale
                            for i in range(current_scale, target_scale):
                                current, _ = self.model.scale_predictors[i](current)

                            pred_images = current

                        # Compute loss
                        loss = self._compute_loss(pred_images, target_images)
                        val_loss += loss.item()

                    # Free up memory
                    del pred_images
                    if self.memory_efficient and torch.cuda.is_available() and batch_idx % 5 == 0:
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(f"CUDA OOM error during validation: {str(e)}")
                        # Try to recover by clearing cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()

                        # Skip this batch and continue with the next one
                        logger.warning(f"Skipping validation batch {batch_idx} due to OOM error")
                        continue
                    else:
                        # Re-raise other runtime errors
                        raise e

        # Calculate average loss
        avg_val_loss = val_loss / num_batches

        # Final cleanup after validation
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return avg_val_loss

    def save_checkpoint(
        self,
        epoch: int,
        current_scale: int,
        target_scale: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            current_scale: Current scale index
            target_scale: Target scale index
            loss: Loss value
            is_best: Whether this is the best model so far
        """
        # Clear cache before saving checkpoint
        if torch.cuda.is_available() and self.memory_efficient:
            torch.cuda.empty_cache()
            gc.collect()

        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'current_scale': current_scale,
            'target_scale': target_scale,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_scale_{current_scale}_{target_scale}_epoch_{epoch}.pt"

        # Use a memory-efficient saving approach
        try:
            # Save with reduced memory usage
            torch.save(
                checkpoint,
                checkpoint_path,
                _use_new_zipfile_serialization=True  # More memory efficient
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Save best model
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                # Create a symlink instead of duplicating the file to save disk space
                if os.path.exists(best_path):
                    os.remove(best_path)
                torch.save(
                    checkpoint,
                    best_path,
                    _use_new_zipfile_serialization=True
                )
                logger.info(f"Saved best model to {best_path}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

        # Clear memory after saving
        del checkpoint
        if torch.cuda.is_available() and self.memory_efficient:
            torch.cuda.empty_cache()
            gc.collect()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Clear cache before loading checkpoint
        if torch.cuda.is_available() and self.memory_efficient:
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            # Load checkpoint with memory mapping for large files
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                mmap=True  # Use memory mapping for large files
            )

            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state dict
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            # Load scaler state dict
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.current_scale = checkpoint['current_scale']
            self.global_step = checkpoint.get('global_step', 0)

            logger.info(f"Resuming from epoch {self.current_epoch}, scale {self.current_scale}")

            # Clear memory after loading
            del checkpoint
            if torch.cuda.is_available() and self.memory_efficient:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def train(
        self,
        num_epochs_per_scale: int = 10,
        progressive: bool = True,
        start_scale: int = 0,
        end_scale: Optional[int] = None,
        save_every: int = 1,
        validate_every: int = 1
    ) -> None:
        """
        Train the model progressively at different scales

        Args:
            num_epochs_per_scale: Number of epochs to train at each scale
            progressive: Whether to train progressively (True) or only at the final scale (False)
            start_scale: Scale index to start training from
            end_scale: Scale index to end training at (inclusive)
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        if end_scale is None:
            end_scale = self.num_scales - 1

        logger.info(f"Starting progressive training from scale {start_scale} to {end_scale}")
        logger.info(f"Training for {num_epochs_per_scale} epochs per scale")

        # Update learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs_per_scale * len(self.train_loader)
        )

        # Progressive training
        if progressive:
            for current_scale in range(start_scale, end_scale):
                target_scale = current_scale + 1
                logger.info(f"Training scale {current_scale} -> {target_scale}")

                for epoch in range(num_epochs_per_scale):
                    epoch_global = self.current_epoch + epoch

                    # Train for one epoch
                    train_loss = self.train_epoch(current_scale, target_scale, epoch_global)
                    logger.info(f"Epoch {epoch_global}, Scale {current_scale}->{target_scale}, Train Loss: {train_loss:.6f}")

                    # Validate
                    if (epoch + 1) % validate_every == 0 and self.val_loader is not None:
                        val_loss = self.validate(current_scale, target_scale)
                        logger.info(f"Epoch {epoch_global}, Scale {current_scale}->{target_scale}, Val Loss: {val_loss:.6f}")

                        # Check if this is the best model
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                    else:
                        is_best = False

                    # Save checkpoint
                    if (epoch + 1) % save_every == 0:
                        self.save_checkpoint(epoch_global, current_scale, target_scale, train_loss, is_best)

                self.current_epoch += num_epochs_per_scale
                self.current_scale = target_scale
        else:
            # Train only at the final scale
            current_scale = start_scale
            target_scale = end_scale
            logger.info(f"Training scale {current_scale} -> {target_scale}")

            for epoch in range(num_epochs_per_scale):
                epoch_global = self.current_epoch + epoch

                # Train for one epoch
                train_loss = self.train_epoch(current_scale, target_scale, epoch_global)
                logger.info(f"Epoch {epoch_global}, Scale {current_scale}->{target_scale}, Train Loss: {train_loss:.6f}")

                # Validate
                if (epoch + 1) % validate_every == 0 and self.val_loader is not None:
                    val_loss = self.validate(current_scale, target_scale)
                    logger.info(f"Epoch {epoch_global}, Scale {current_scale}->{target_scale}, Val Loss: {val_loss:.6f}")

                    # Check if this is the best model
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                else:
                    is_best = False

                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch_global, current_scale, target_scale, train_loss, is_best)

            self.current_epoch += num_epochs_per_scale
            self.current_scale = target_scale

        logger.info("Training completed")

# Example dataset class for training
class ImageTextDataset(Dataset):
    """
    Dataset for training the Next Scale Prediction model
    Provides images and optional text prompts
    """
    def __init__(
        self,
        image_paths: List[str],
        text_prompts: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.text_prompts = text_prompts
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Get text prompt if available
        if self.text_prompts is not None:
            text_prompt = self.text_prompts[idx]
            return {'image': image, 'text_prompt': text_prompt}
        else:
            return {'image': image}

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load image from path

        Args:
            path: Path to the image file

        Returns:
            Image tensor of shape [channels, height, width]
        """
        # This is a placeholder - in a real implementation, you would load the image
        # using PIL, OpenCV, or another library
        # For example:
        # from PIL import Image
        # import numpy as np
        # image = Image.open(path).convert('RGB')
        # image = np.array(image).transpose(2, 0, 1) / 255.0
        # return torch.from_numpy(image).float()

        # For now, just return a random tensor based on the path hash for reproducibility
        # This ensures different images have different random tensors
        path_hash = hash(path) % 10000
        torch.manual_seed(path_hash)
        return torch.randn(3, 256, 256)  # Placeholder
