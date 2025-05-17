import torch
import argparse
import logging
from pathlib import Path
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
from typing import List

from nfp.model import create_model_and_vae
from nfp.trainer import ProgressiveTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFolder(Dataset):
	"""Dataset for loading images from a folder"""
	def __init__(self, folder_path: str, transform=None):
		self.folder_path = folder_path
		self.transform = transform
		self.image_paths = self._get_image_paths()

	def _get_image_paths(self) -> List[str]:
		"""Get all image paths in the folder"""
		extensions = ['*.jpg', '*.jpeg', '*.png']
		image_paths = []

		for ext in extensions:
			image_paths.extend(glob.glob(os.path.join(self.folder_path, ext)))
			image_paths.extend(glob.glob(os.path.join(self.folder_path, '**', ext), recursive=True))

		filtered = []
		for path in image_paths:
			try:
				Image.open(path).convert('RGB')
				filtered.append(path)
			except Exception as e:
				logger.warning(f"Skipping {path} due to error: {e}")

		return sorted(filtered)

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, idx: int):
		image_path = self.image_paths[idx]
		image = Image.open(image_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return {'image': image}

def main():
	parser = argparse.ArgumentParser(description="Train Next Scale Prediction model")
	parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
	parser.add_argument("--val_dir", type=str, default=None, help="Directory containing validation images")
	parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
	parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
	parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
	parser.add_argument("--initial_size", type=int, default=16, help="Initial size of noise latent")
	parser.add_argument("--target_size", type=int, default=1024, help="Target image size")
	parser.add_argument("--scale_factor", type=int, default=2, help="Factor by which to scale at each step")
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--num_epochs_per_scale", type=int, default=10, help="Number of epochs to train at each scale")
	parser.add_argument("--progressive", action="store_true", help="Train progressively at each scale")
	parser.add_argument("--start_scale", type=int, default=0, help="Scale index to start training from")
	parser.add_argument("--end_scale", type=int, default=None, help="Scale index to end training at (inclusive)")
	parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
	parser.add_argument("--validate_every", type=int, default=1, help="Validate every N epochs")
	parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
	parser.add_argument("--fp16", type=lambda x: x.lower() == 'true', default=False,
						help="Use mixed precision training (true/false)")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
						help="Device to run training on")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
	parser.add_argument("--memory_efficient", action="store_true", help="Enable memory optimization techniques")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
						help="Number of steps to accumulate gradients before updating weights")
	parser.add_argument("--prefetch_factor", type=int, default=2,
						help="Number of batches to prefetch per worker")
	parser.add_argument("--pin_memory", action="store_true", default=True,
						help="Use pinned memory for faster data transfer to GPU")
	parser.add_argument("--persistent_workers", action="store_true", default=True,
						help="Keep worker processes alive between epochs")
	parser.add_argument("--cpu_offload", action="store_true",
						help="Offload some operations to CPU to save GPU memory")
	args = parser.parse_args()

	# Create output directories
	output_dir = Path(args.output_dir)
	checkpoint_dir = Path(args.checkpoint_dir)
	log_dir = Path(args.log_dir)

	output_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	log_dir.mkdir(parents=True, exist_ok=True)

	# Set up data transforms
	transform = transforms.Compose([
		transforms.Resize((args.target_size, args.target_size)),
		transforms.ToTensor()
	])

	# Create datasets
	train_dataset = ImageFolder(args.data_dir, transform=transform)
	logger.info(f"Created training dataset with {len(train_dataset)} images")

	if args.val_dir:
		val_dataset = ImageFolder(args.val_dir, transform=transform)
		logger.info(f"Created validation dataset with {len(val_dataset)} images")
	else:
		val_dataset = None
		logger.info("No validation dataset provided")

	# Create model and VAE
	logger.info("Creating model and VAE...")
	model, vae = create_model_and_vae(
		initial_size=args.initial_size,
		target_size=args.target_size,
		scale_factor=args.scale_factor,
		device=args.device,
		dtype=torch.float32  # Explicitly use float32 to avoid dtype mismatches
	)

	# Create trainer with memory optimization
	trainer = ProgressiveTrainer(
		model=model,
		vae=vae,
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		batch_size=args.batch_size,
		learning_rate=args.learning_rate,
		num_workers=args.num_workers,
		device=args.device,
		checkpoint_dir=args.checkpoint_dir,
		log_dir=args.log_dir,
		fp16=args.fp16,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		memory_efficient=args.memory_efficient,
		prefetch_factor=args.prefetch_factor,
		pin_memory=args.pin_memory,
		persistent_workers=args.persistent_workers,
		cpu_offload=args.cpu_offload
	)

	# Resume from checkpoint if provided
	if args.resume_from:
		trainer.load_checkpoint(args.resume_from)

	# Check if start_scale and end_scale are valid
	num_scales = len(trainer.scale_sizes)
	logger.info(f"Model has {num_scales} scales: {trainer.scale_sizes}")

	start_scale = min(args.start_scale, num_scales - 2)
	end_scale = args.end_scale if args.end_scale is not None else num_scales - 1
	end_scale = min(end_scale, num_scales - 1)

	logger.info(f"Training from scale {start_scale} to scale {end_scale}")

	# Train the model
	trainer.train(
		num_epochs_per_scale=args.num_epochs_per_scale,
		progressive=args.progressive,
		start_scale=start_scale,
		end_scale=end_scale,
		save_every=args.save_every,
		validate_every=args.validate_every
	)

	logger.info("Training completed")

if __name__ == "__main__":
	main()
