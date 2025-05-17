@echo off
REM Inference script for NextFramePredict model

REM Set environment variables for better memory management
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_LAUNCH_BLOCKING=0

REM Run inference with the latest checkpoint
py inference.py ^
    --checkpoint "checkpoints/checkpoint_scale_1_2_epoch_9.pt" ^
    --output_dir "generated_images" ^
    --initial_size 16 ^
    --target_size 64 ^
    --scale_factor 2 ^
    --num_images 4 ^
    --prompts "a beautiful landscape with mountains and a lake" "a futuristic city skyline at night" ^
    --seed 42 ^
    --save_intermediates

REM Note: Adjust the following parameters based on your needs:
REM --checkpoint: Path to the checkpoint file (use "best_model.pt" or a specific checkpoint)
REM --target_size: Should match the size used during training
REM --num_images: Number of images to generate per prompt
REM --prompts: List of text prompts (or use --prompts_file to load from a file)
