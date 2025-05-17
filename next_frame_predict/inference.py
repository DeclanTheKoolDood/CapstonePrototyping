import torch
import argparse
import logging
from pathlib import Path
import os
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import random
import time

from nfp.model import create_model_and_vae, NextScalePredictionModel
from nfp.encoders import SDXLVAEWrapper
from nfp.trainer import ProgressiveTrainer
from nfp.utils import save_image, create_grid

def decode_with_vae(
	vae: SDXLVAEWrapper,
	latents: torch.Tensor,
	expected_channels: int = 4
) -> torch.Tensor:
	"""
	Decode latents with VAE, handling channel mismatches

	Args:
		vae: SDXL VAE model for decoding
		latents: Latents to decode
		expected_channels: Expected number of channels in the VAE input

	Returns:
		Decoded images
	"""
	# Check if we need to adjust channels
	if latents.shape[1] != expected_channels:
		logger.info(f"Adjusting channels from {latents.shape[1]} to {expected_channels}")

		# Create a new tensor with the expected number of channels
		adjusted_latents = torch.zeros(
			latents.shape[0], expected_channels, latents.shape[2], latents.shape[3],
			device=latents.device, dtype=latents.dtype
		)

		# Copy the available channels
		min_channels = min(latents.shape[1], expected_channels)
		adjusted_latents[:, :min_channels, :, :] = latents[:, :min_channels, :, :]

		# Use the adjusted latents
		latents = adjusted_latents

	# Decode with VAE
	return vae.decode(latents)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_checkpoint(
	checkpoint_path: str,
	initial_size: int,
	target_size: int,
	scale_factor: int,
	device: str
) -> Tuple[NextScalePredictionModel, SDXLVAEWrapper]:
	"""
	Load model and VAE from checkpoint

	Args:
		checkpoint_path: Path to checkpoint file
		initial_size: Initial size of noise latent
		target_size: Target image size
		scale_factor: Factor by which to scale at each step
		device: Device to run inference on

	Returns:
		Tuple containing:
		- Loaded NextScalePredictionModel
		- Initialized SDXLVAEWrapper
	"""
	logger.info(f"Loading checkpoint from {checkpoint_path}")

	# Create model and VAE
	model, vae = create_model_and_vae(
		initial_size=initial_size,
		target_size=target_size,
		scale_factor=scale_factor,
		device=device,
		dtype=torch.float32  # Always use float32 to avoid dtype mismatches
	)

	# Load checkpoint
	checkpoint = torch.load(checkpoint_path, map_location=device)

	# Load model state dict
	model.load_state_dict(checkpoint['model_state_dict'])

	# Set model to eval mode
	model.eval()

	logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, scale {checkpoint['current_scale']}->{checkpoint['target_scale']}")

	return model, vae

def generate_images(
	model: NextScalePredictionModel,
	vae: SDXLVAEWrapper,
	text_prompts: List[str],
	num_images: int,
	initial_size: int,
	output_dir: Path,
	seed: Optional[int] = None,
	save_intermediates: bool = False
) -> List[torch.Tensor]:
	"""
	Generate images using the model

	Args:
		model: The trained NextScalePredictionModel
		vae: SDXL VAE model for final decoding
		text_prompts: List of text prompts for conditioning
		num_images: Number of images to generate per prompt
		initial_size: Initial size of noise latent
		output_dir: Directory to save generated images
		seed: Optional random seed for reproducibility
		save_intermediates: Whether to save intermediate outputs

	Returns:
		List of generated images
	"""
	device = model.device
	all_images = []

	# Create output directory
	output_dir.mkdir(parents=True, exist_ok=True)

	# Generate images for each prompt
	for prompt_idx, prompt in enumerate(text_prompts):
		logger.info(f"Generating images for prompt: '{prompt}'")

		# Set seed for reproducibility
		if seed is not None:
			current_seed = seed + prompt_idx
			torch.manual_seed(current_seed)
			random.seed(current_seed)
			np.random.seed(current_seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed_all(current_seed)

		# Generate without using the built-in VAE decoding in generate_image
		with torch.no_grad():
			# Create random noise as starting point
			noise = torch.randn(num_images, 3, initial_size, initial_size,
							   device=device, dtype=torch.float32)

			# Generate latents
			logger.info(f"Generating latents for prompt: '{prompt}'")
			latents, intermediates = model(noise, [prompt] * num_images, return_intermediates=True)

			# Decode with our custom VAE decoder that handles channel mismatches
			logger.info("Decoding latents with SDXL VAE")
			images = decode_with_vae(vae, latents)

			# Process intermediates to make them viewable
			processed_intermediates = []
			for intermediate in intermediates:
				# For each intermediate output, we need to ensure it's in a viewable format
				# This might involve normalizing or other processing
				processed_intermediate = torch.clamp(intermediate, 0, 1)
				processed_intermediates.append(processed_intermediate)

			intermediates = processed_intermediates

		# Save images
		prompt_dir = output_dir / f"prompt_{prompt_idx}"
		prompt_dir.mkdir(parents=True, exist_ok=True)

		# Save prompt to text file
		with open(prompt_dir / "prompt.txt", "w") as f:
			f.write(prompt)

		# Save individual images
		for i in range(num_images):
			image_path = prompt_dir / f"image_{i}.png"
			save_image(images[i], str(image_path))
			all_images.append(images[i])

		# Save grid of images
		if num_images > 1:
			grid = create_grid(images, nrow=min(4, num_images))
			grid_path = prompt_dir / "grid.png"
			save_image(grid, str(grid_path))

		# Save intermediates if requested
		if save_intermediates:
			intermediates_dir = prompt_dir / "intermediates"
			intermediates_dir.mkdir(parents=True, exist_ok=True)

			for i, intermediate in enumerate(intermediates):
				for j in range(num_images):
					intermediate_path = intermediates_dir / f"scale_{i}_image_{j}.png"
					save_image(intermediate[j], str(intermediate_path))

				# Save grid of intermediates
				if num_images > 1:
					grid = create_grid(intermediate, nrow=min(4, num_images))
					grid_path = intermediates_dir / f"scale_{i}_grid.png"
					save_image(grid, str(grid_path))

	return all_images

def main():
	parser = argparse.ArgumentParser(description="Generate images using trained NextFramePredict model")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
	parser.add_argument("--output_dir", type=str, default="generated_images", help="Directory to save generated images")
	parser.add_argument("--initial_size", type=int, default=16, help="Initial size of noise latent")
	parser.add_argument("--target_size", type=int, default=64, help="Target image size")
	parser.add_argument("--scale_factor", type=int, default=2, help="Factor by which to scale at each step")
	parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate per prompt")
	parser.add_argument("--prompts", type=str, nargs="+", default=["a beautiful landscape with mountains and a lake"],
						help="Text prompts for conditioning")
	parser.add_argument("--prompts_file", type=str, default=None, help="File containing text prompts (one per line)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
						help="Device to run inference on")
	parser.add_argument("--save_intermediates", action="store_true", help="Save intermediate outputs")
	args = parser.parse_args()

	# Create output directory
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Load text prompts from file if provided
	if args.prompts_file is not None and os.path.exists(args.prompts_file):
		with open(args.prompts_file, "r") as f:
			prompts = [line.strip() for line in f.readlines() if line.strip()]
		logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
	else:
		prompts = args.prompts

	# Load model and VAE from checkpoint
	model, vae = load_checkpoint(
		checkpoint_path=args.checkpoint,
		initial_size=args.initial_size,
		target_size=args.target_size,
		scale_factor=args.scale_factor,
		device=args.device
	)

	# Generate images
	start_time = time.time()
	generate_images(
		model=model,
		vae=vae,
		text_prompts=prompts,
		num_images=args.num_images,
		initial_size=args.initial_size,
		output_dir=output_dir,
		seed=args.seed,
		save_intermediates=args.save_intermediates
	)
	end_time = time.time()

	logger.info(f"Generated {len(prompts) * args.num_images} images in {end_time - start_time:.2f} seconds")
	logger.info(f"Images saved to {output_dir}")

if __name__ == "__main__":
	main()
