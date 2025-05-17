import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from PIL import Image
import io
import base64
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_image(
	image: torch.Tensor,
	target_size: int,
	mode: str = 'bilinear'
) -> torch.Tensor:
	"""
	Resize image to target size using interpolation

	Args:
		image: Image tensor of shape [batch_size, channels, height, width]
		target_size: Target size for height and width
		mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')

	Returns:
		Resized image tensor of shape [batch_size, channels, target_size, target_size]
	"""
	if image.shape[2] == target_size and image.shape[3] == target_size:
		return image

	return F.interpolate(
		image,
		size=(target_size, target_size),
		mode=mode,
		align_corners=False if mode != 'nearest' else None
	)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
	"""
	Convert a tensor to a PIL Image

	Args:
		tensor: Image tensor of shape [channels, height, width] in range [0, 1]

	Returns:
		PIL Image
	"""
	# Convert to numpy array
	array = tensor.cpu().detach().numpy()

	# Transpose from [channels, height, width] to [height, width, channels]
	array = array.transpose(1, 2, 0)

	# Scale from [0, 1] to [0, 255]
	array = (array * 255).astype(np.uint8)

	# Convert to PIL Image
	return Image.fromarray(array)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
	"""
	Convert a PIL Image to a tensor

	Args:
		image: PIL Image

	Returns:
		Image tensor of shape [channels, height, width] in range [0, 1]
	"""
	# Convert to numpy array
	array = np.array(image)

	# Ensure RGB format
	if len(array.shape) == 2:  # Grayscale
		array = np.stack([array] * 3, axis=-1)
	elif array.shape[2] == 4:  # RGBA
		array = array[:, :, :3]

	# Transpose from [height, width, channels] to [channels, height, width]
	array = array.transpose(2, 0, 1)

	# Scale from [0, 255] to [0, 1]
	array = array.astype(np.float32) / 255.0

	# Convert to tensor
	return torch.from_numpy(array)

def save_image(
	tensor: torch.Tensor,
	path: str,
	normalize: bool = False
) -> None:
	"""
	Save a tensor as an image

	Args:
		tensor: Image tensor of shape [channels, height, width]
		path: Path to save the image
		normalize: Whether to normalize the image to [0, 1]
	"""
	# Ensure the tensor is on CPU
	tensor = tensor.cpu().detach()

	# Normalize if requested
	if normalize:
		tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

	# Clamp to [0, 1]
	tensor = torch.clamp(tensor, 0, 1)

	# Convert to PIL Image and save
	image = tensor_to_pil(tensor)
	image.save(path)

	logger.info(f"Saved image to {path}")

def tensor_to_base64(
	tensor: torch.Tensor,
	format: str = 'PNG'
) -> str:
	"""
	Convert a tensor to a base64-encoded image

	Args:
		tensor: Image tensor of shape [channels, height, width] in range [0, 1]
		format: Image format ('PNG', 'JPEG', etc.)

	Returns:
		Base64-encoded image string
	"""
	# Convert to PIL Image
	image = tensor_to_pil(tensor)

	# Save to bytes buffer
	buffer = io.BytesIO()
	image.save(buffer, format=format)
	buffer.seek(0)

	# Encode as base64
	return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_grid(
	tensors: List[torch.Tensor],
	nrow: int = 8,
	padding: int = 2
) -> torch.Tensor:
	"""
	Create a grid of images

	Args:
		tensors: List of image tensors, each of shape [channels, height, width]
		nrow: Number of images per row
		padding: Padding between images

	Returns:
		Grid tensor of shape [channels, height*nrow, width*ncol]
	"""
	# Get dimensions
	n = len(tensors)
	if n == 0:
		return torch.empty(0)

	ncol = (n + nrow - 1) // nrow

	# Ensure all tensors have the same shape
	shapes = set(tuple(t.shape) for t in tensors)
	if len(shapes) > 1:
		# Resize all tensors to the same shape (use the first tensor's shape)
		target_shape = tensors[0].shape
		tensors = [
			F.interpolate(t.unsqueeze(0), size=target_shape[1:], mode='bilinear', align_corners=False).squeeze(0)
			if t.shape != target_shape else t
			for t in tensors
		]

	# Get dimensions
	channels, height, width = tensors[0].shape

	# Create empty grid
	grid = torch.zeros(
		channels,
		height * nrow + padding * (nrow - 1),
		width * ncol + padding * (ncol - 1),
		device=tensors[0].device
	)

	# Fill grid
	for i, tensor in enumerate(tensors):
		row = i // ncol
		col = i % ncol
		grid[
			:,
			row * (height + padding) : row * (height + padding) + height,
			col * (width + padding) : col * (width + padding) + width
		] = tensor

	return grid

def load_image(
	path: str,
	target_size: Optional[int] = None,
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
	"""
	Load an image from path

	Args:
		path: Path to the image
		target_size: Optional target size for resizing
		device: Device to load the tensor to

	Returns:
		Image tensor of shape [1, channels, height, width] in range [0, 1]
	"""
	# Load image
	image = Image.open(path).convert('RGB')

	# Resize if requested
	if target_size is not None:
		image = image.resize((target_size, target_size), Image.LANCZOS)

	# Convert to tensor
	tensor = pil_to_tensor(image).unsqueeze(0).to(device)

	return tensor
