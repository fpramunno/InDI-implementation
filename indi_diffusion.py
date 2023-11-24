# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:40:10 2023

@author: pio-r
"""

import torch
from tqdm import tqdm
import torch.nn as nn
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Indi_cond:
    """
    The Indi_cond class implements the diffusion-like process for image generation of the InDI paper.
    It allows sampling new images based on a given model and input parameters.
    
    Attributes:
        noise_steps (int): Number of timesteps in the diffusion process.
        img_size (int): Size of the images.
        img_channel (int): Number of channels in the images.
        device (str): Device to use for computations (e.g., 'cuda' for GPU).
    """
    def __init__(self, img_size=256, img_channel=1, device="cuda"):
        self.img_channel = img_channel
        self.img_size = img_size
        self.device = device

    def sample_timesteps(self, n):
        """
        Samples random timesteps for the diffusion process.
    
        Args:
            n (int): Number of timesteps to sample.
    
        Returns:
            torch.Tensor: A tensor of randomly sampled timesteps.
        """
        return torch.randint(size=(n,))

    def sample(self, model, x, n, steps, labels):
        logging.info(f"Sampling {n} new images....")
        model.eval() # evaluation mode
        with torch.no_grad():
            for t in tqdm(torch.linspace(1, 0, steps+1, device='cuda')[:-1]):
                if x.shape[0] > 1:
                    t = t[:, None, None, None]
                else:
                    t = t
                predicted_peak = model(x, labels, t)
                fct = 1/(steps*t)
                x = (1-fct) * x + fct * predicted_peak
                
            
            model.train() # it goes back to training mode
            x = (x.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
            x = (x * 255).type(torch.uint8) # to bring in valid pixel range
            return x
    
mse = nn.MSELoss()

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Create a function that calculates the PSNR between 2 images.

    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Args:
        input: the input image with arbitrary shape :math:`(*)`.
        labels: the labels image with arbitrary shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(20.0000)

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val**2 / mse(input, target))