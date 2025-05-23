

# TODO

- explain what this is
- what the features are
- what is not implemented
- future ideas
- basic setup, if applicable
- tools used
- prompt tips and tricks

## Lightning Image Diffusion

Lightning image diffusion is an experimental model mostly made by AI to test the agentic capabilities of AUGMENT and Co-Pilot in creating an AI model with specified features.
It produced this code in a step-by-step nature, listed in the *Prompting Tips and Tricks* section and successfully had a forward pass.
The model is NOT trained due to a lack of dataset and up-to-date training methods as this model used a simplified trainer.

## Prompting Tips and Tricks

These steps are essentially the same for the next_frame_predict model.

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
