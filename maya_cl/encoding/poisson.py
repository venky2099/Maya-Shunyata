# poisson.py — rate-coded Poisson spike encoder for CIFAR-10 images

import torch


class PoissonEncoder:
    """
    Converts a normalised image tensor [B, C, H, W] in [0,1]
    into a binary spike tensor [T, B, C, H, W] via Poisson rate coding.
    Pixel intensity = firing probability at each timestep.
    """

    def __init__(self, t_steps: int):
        self.t_steps = t_steps  # number of timesteps per forward pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], values in [0, 1]
        # returns: [T, B, C, H, W], binary spikes
        x = x.clamp(0.0, 1.0)
        spikes = torch.rand(self.t_steps, *x.shape, device=x.device)
        return (spikes < x.unsqueeze(0)).float()