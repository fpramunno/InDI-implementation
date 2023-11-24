# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:28:21 2023

@author: pio-r
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    This class implements methods to update the moving average of model parameters over time,
    which can be useful for smoothing the parameters in training.

    Attributes:
        beta (float): The decay rate for the moving average.
        step (int): The current step of the model update.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Updates the moving average model's parameters.
    
        Args:
            ma_model (torch.nn.Module): The model to update with the moving average.
            current_model (torch.nn.Module): The current model providing new parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    """
    A Self-Attention module implementing multi-headed attention mechanism.

    This module applies a multi-head attention mechanism on the input feature map,
    followed by layer normalization and a feedforward neural network.

    Attributes:
        channels (int): The number of channels in the input.
        size (int): The size of each attention head.
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer
    
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # projection
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class IndiUnet(nn.Module):
    def __init__(self, c_in=1, c_out=1, image_size=64, time_dim=256, device='cuda', latent=False, true_img_size=64, num_classes=None):
        super(IndiUnet, self).__init__()

        # Encoder
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(c_in, self.image_size) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size*2) # input and output channels
        # self.sa1 = SelfAttention(self.image_size*2,int( self.true_img_size/2)) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size*2, self.image_size*4)
        # self.sa2 = SelfAttention(self.image_size*4, int(self.true_img_size/4))
        self.down3 = Down(self.image_size*4, self.image_size*4)
        # self.sa3 = SelfAttention(self.image_size*4, int(self.true_img_size/8))
        
        # Bootleneck
        self.bot1 = DoubleConv(self.image_size*4, self.image_size*8)
        self.bot2 = DoubleConv(self.image_size*8, self.image_size*8)
        self.bot3 = DoubleConv(self.image_size*8, self.image_size*4)
        
        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size*8, self.image_size*2)
        # self.sa4 = SelfAttention(self.image_size*2, int(self.true_img_size/4))
        self.up2 = Up(self.image_size*4, self.image_size)
        # self.sa5 = SelfAttention(self.image_size, int(self.true_img_size/2))
        self.up3 = Up(self.image_size*2, self.image_size)
        # self.sa6 = SelfAttention(self.image_size, self.true_img_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1) # projecting back to the output channel dimensions
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        if latent == True:
            self.latent = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256)).to(device)    
  
    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc 

    def forward(self, x, lab, t):
        # Pass the source image through the encoder network
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode

        
        if lab is not None:
            t += self.label_emb(lab)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t) # We note that upsampling box that in the skip connections from encoder 
        # x = self.sa4(x)
        x = self.up2(x, x2, t)
        # x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)

        return output