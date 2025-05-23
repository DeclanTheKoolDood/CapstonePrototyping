# Next Scale Prediction (NSP)

Next Scale Prediction is a progressive image generation model that generates high-resolution images by iteratively predicting larger scales from smaller ones.

## Features

1. **Progressive Scale Generation**: Starts with a small noise latent and progressively predicts larger scales
2. **Text Conditioning**: Controls generation through text prompts using CLIP text encoder
3. **Residual Dense Blocks**: Improves gradient flow and feature reuse
4. **Multi-Query / Grouped Attention**: Reduces memory usage by sharing key/value pairs
5. **Adaptive Layer Norm (FiLM)**: Applies conditioning at each scale
6. **Residual Connections for Upscaling**: Uses bilinear interpolation and predicts residuals
7. **SDXL VAE Integration**: Uses SDXL1.0 VAE for high-quality image encoding/decoding
8. **Mixed Precision Support**: Consistent dtype handling for both float32 and float16 inputs

## Prompting Tips and Tricks

These steps are essentially the same for the lightning_image_diffusion model.

#### **Context Creation**
This one is similar to other medium and larger projects where creating context is incredibly important.
I listed all the features I wanted the model to have, like "tiled image encoding with positional encoding and overlap", "low memory usage", "fast convergence", "fast performance", etc.
I also asked questions like "how can I make this model converge faster?", "how can I train this at incredible speeds", etc.
I took down a dot point list of all these features, ideas and concepts, then got the AI to format it and provide concise descriptions.

#### **Step-By-Step Implementation**
I got the agentic coder to implement the model step-by-step, focusing on individual components before combining them together.
I got it to make the AI model individual layers first, things like the text encoder, the VAE (which encodes images to a compressed "latent"), and individual model layers like the VAE tiling.
I got the AI to put together the actual model, the diffusion model architecture, with all the individual layers that were made and additional layers that are common with the diffusion architecture.
I then did the dataset loader, the trainer code, and testing one forward pass with an image and fixed any issues with that. The forward pass essentially marks the model as "testable" and "trainable" as images can be fully passed through the model.

## Installation

```bash
git clone https://github.com/yourusername/NextFramePredict.git
cd NextFramePredict
pip install -r requirements.txt
```

## Usage

### Generate Images

```bash
python example.py --text_prompt "a beautiful landscape with mountains and a lake" --output_dir outputs
```

### Parameters

- `--output_dir`: Directory to save generated images (default: "outputs")
- `--initial_size`: Initial size of noise latent (default: 16)
- `--target_size`: Target image size (default: 1024)
- `--scale_factor`: Factor by which to scale at each step (default: 2)
- `--batch_size`: Number of images to generate (default: 1)
- `--text_prompt`: Text prompt for conditioning (default: "a beautiful landscape with mountains and a lake")
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to run generation on (default: "cuda" if available, else "cpu")
- `--save_intermediates`: Save intermediate outputs at each scale

## Model Architecture

The Next Scale Prediction model consists of the following components:

1. **CLIP Text Encoder**: Encodes text prompts for conditioning
2. **Scale Predictors**: Predict the next scale from the current scale
3. **Residual Dense Blocks**: Improve gradient flow and feature reuse
4. **Grouped Self-Attention**: Reduces memory usage
5. **Adaptive Layer Norm**: Applies conditioning at each scale
6. **SDXL VAE**: Encodes/decodes images to/from latent space

## Training

The model can be trained progressively at different scales, using interpolation to resize images between scales.

### Progressive Training

```bash
python train_example.py --data_dir path/to/images --progressive --num_epochs_per_scale 10
```

### Parameters

- `--data_dir`: Directory containing training images
- `--val_dir`: Directory containing validation images (optional)
- `--output_dir`: Directory to save outputs (default: "outputs")
- `--checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")
- `--log_dir`: Directory to save logs (default: "logs")
- `--initial_size`: Initial size of noise latent (default: 16)
- `--target_size`: Target image size (default: 1024)
- `--scale_factor`: Factor by which to scale at each step (default: 2)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs_per_scale`: Number of epochs to train at each scale (default: 10)
- `--progressive`: Train progressively at each scale (flag)
- `--start_scale`: Scale index to start training from (default: 0)
- `--end_scale`: Scale index to end training at (inclusive, default: None)
- `--save_every`: Save checkpoint every N epochs (default: 1)
- `--validate_every`: Validate every N epochs (default: 1)
- `--resume_from`: Resume training from checkpoint (default: None)
- `--fp16`: Use mixed precision training (flag)
- `--device`: Device to run training on (default: "cuda" if available, else "cpu")
- `--num_workers`: Number of workers for data loading (default: 4)
