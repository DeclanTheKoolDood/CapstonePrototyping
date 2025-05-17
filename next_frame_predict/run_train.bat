@echo off
REM Memory-optimized training script for NextScalePrediction model

REM Set environment variables for better memory management
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_LAUNCH_BLOCKING=0

REM Run training with memory optimization
py train_example.py ^
    --data_dir C:\\Users\\Declan\\Pictures\\dataset ^
    --batch_size 6 ^
    --initial_size 16 ^
    --target_size 64 ^
    --fp16 false ^
    --scale_factor 2 ^
    --learning_rate 1e-4 ^
    --num_epochs_per_scale 5 ^
    --progressive ^
    --start_scale 0 ^
    --save_every 1 ^
    --validate_every 1 ^
    --num_workers 2 ^
    --memory_efficient ^
    --gradient_accumulation_steps 4 ^
    --prefetch_factor 2 ^
    --pin_memory ^
    --persistent_workers ^
    --cpu_offload

REM Note: Adjust the following parameters based on your GPU memory:
REM --batch_size: Lower for less memory usage (1-2 recommended)
REM --target_size: Lower for less memory usage (128-256 recommended for <12GB VRAM)
REM --gradient_accumulation_steps: Higher for less memory usage (4-8 recommended)
