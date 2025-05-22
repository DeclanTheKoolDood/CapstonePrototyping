

# TODO

- explain what this is
- what the features are
- what is not implemented
- future ideas
- basic setup, if applicable
- tools used
- prompt tips and tricks


# Context Notes

I found that, especially with technical projects, creating context for the project is essential to get good outputs.

For example, for the lightning diffusion model, I created a notepad file prompt provided below with all the features and components i wanted,
then i fed that into the copilot and asked it to improve it and format it, repeated that a few times, then incrementally implemented it.

Example prompt when doing the "training script" for lightning diffusion:
```
Use "DPM-Solver++" schedular for fast diffusion.
Use mixed precision training.
Use gradient accumulation.
Use gradient checkpointing.
Use EMA weights.
Use efficient data loading.
Use Learning Rate Scheduling and Warmup.
Use  Early Stopping and Checkpointing.
Use Smaller Model/Parameter Sharing.
Use Profiling and add profilers in the code with options to toggle so the training can be profiled.
Use "Adafactor transformers.optimization.Adafactor".
Use "Patch-based MSELoss on the VAE latent and original/output image with high precision" to get as close to the encoded image as possible.

Use Data Augmentation Techniques & Ideas:
- Geometric Augmentations: Random crop, resize, horizontal/vertical flip, rotation, affine transforms.
- Color Augmentations: Color jitter (brightness, contrast, saturation, hue), grayscale, random gamma.
- Random Erasing: Randomly mask out small patches of the image.
- Gaussian/Poisson Noise: Add random noise to simulate sensor or compression artifacts.
- GridMask: Mask out regions in a grid pattern for regularization.

Make sure to implement all the above, do it step-by-step focusing on a basic training script first, then applying all features on top.
```
