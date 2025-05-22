import torch
import numpy as np

class DPMSolverPP:
    """
    DPM-Solver++ scheduler for fast diffusion sampling.
    Reference: https://github.com/LuChengTHU/dpm-solver
    """
    def __init__(self, num_train_timesteps=1000, num_inference_steps=10, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=int)[::-1]

    def step(self, model_output, timestep, sample):
        # Simplified single-step update for demonstration
        alpha = self.alphas_cumprod[timestep]
        prev_t = max(timestep - self.num_train_timesteps // self.num_inference_steps, 0)
        prev_alpha = self.alphas_cumprod[prev_t]
        pred_x0 = (sample - np.sqrt(1 - alpha) * model_output) / np.sqrt(alpha)
        dir_xt = np.sqrt(1 - prev_alpha) * model_output
        prev_sample = np.sqrt(prev_alpha) * pred_x0 + dir_xt
        return prev_sample

    def get_timesteps(self):
        return self.timesteps
