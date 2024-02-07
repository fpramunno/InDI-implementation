# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:41:07 2023

@author: pio-r
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Normalize

def save_images(img, path):
    """
    Saves a batch of images to the specified path.

    Args:
        img (torch.Tensor): A batch of images as a tensor.
        path (str): The path where images will be saved.

    This function iterates over each image in the batch, converts it from a tensor to a PIL image,
    and saves it to the specified path.
    """
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        v = Image.fromarray(images)
        V.append(v)
    for value in V:
        value.save(path)

# Data loading

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for the Indi model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--run_name', type=str, default="DDPM_Conditional", help='Name of the run')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=1024, help='Size of the images')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    return parser.parse_args()

args = parse_args()
data_path = args.data_path
args.run_name = "DDPM_Conditional"
args.epochs = 500
args.batch_size = 4
args.image_size = 1024
args.device = "cuda"
args.lr = 3e-4
# Now, use 'data_path' wherever you need to specify the path to the data

from torch.utils.data import Dataset

class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset for loading PNG images.

        Args:
            root_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB if needed

        if self.transform:
            image = self.transform(image)

        return image
        
root_dir = data_path

import torchvision.transforms as transforms

transform = transforms.Compose([Resize((args.image_size, args.image_size)),  # Resize images
                                    transforms.ToTensor(), 
                                    Normalize(mean=(0.5), std=(0.5))])

train_dataset = PNGDataset(root_dir, transform=transform)
val_dataset = PNGDataset(root_dir, transform=transform)


train_data = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=False,
                          pin_memory=True,# pin_memory set to True
                          num_workers=12,
                          prefetch_factor=4,
                          drop_last=False)

val_data = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False,
                          pin_memory=True,# pin_memory set to True
                          num_workers=12,
                          prefetch_factor=4,  # pin_memory set to True
                          drop_last=False)

print('Train loader and Valid loader are up!')



# from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch import optim
import copy
import logging
import torch.nn as nn

# from utils import plot_images, save_images, get_data
from modules import IndiUnet, EMA
from indi_diffusion import psnr
from indi_diffusion import Indi_cond
from torch.utils.tensorboard import SummaryWriter

def setup_logging(run_name):
    """
    Setting up the folders for saving the model and the results

    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


setup_logging(args.run_name)
device = args.device
dataloader = train_data
dataloader_val = val_data
model = IndiUnet(c_in=1, c_out=1, image_size=int(args.image_size), true_img_size=args.image_size).to(device)
diffusion = Indi_cond(img_size=args.image_size, img_channel=1)


optimizer = optim.AdamW(model.parameters(), lr=args.lr)
mse = nn.MSELoss()
logger = SummaryWriter(os.path.join("runs", args.run_name))
l = len(dataloader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)


min_valid_loss = np.inf



from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
for epoch in range(args.epochs):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(train_data)
    model.train()
    train_loss = 0.0
    psnr_train = 0.0
    for i, (dirty, clean) in enumerate(pbar):
        dirty = dirty.float().to(device)
        clean = clean.float().to(device)
        
        labels = None
        t = torch.rand(size=(clean.shape[0],)).to(device)
        
        with autocast():
            fct = t[:, None, None, None]
            transformed_image = (1-fct)*clean + fct*dirty
            predicted_peak = model(transformed_image, labels, t)
            
            loss = mse(clean, predicted_peak)
        
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.step_ema(ema_model, model)

        train_loss += loss.detach().item() * dirty.size(0)
        psnr_train += psnr(predicted_peak, clean, torch.max(predicted_peak)).detach()
        pbar.set_postfix(MSE=loss.item())
        logger.add_scalar("MSE", loss.item(), global_step=epoch * len(pbar) + i)
    
    # Clean up memory before validation
    torch.cuda.empty_cache()

    # Validation step
    valid_loss = 0.0
    psnr_val = 0.0
    pbar_val = tqdm(val_data)
    model.eval()
    with torch.no_grad():
        for i, (dirty, clean) in enumerate(pbar_val):
            dirty = dirty.float().to(device)
            clean = clean.float().to(device)
            
            labels = None
            t = torch.rand(size=(clean.shape[0],)).to(device)
            
            with autocast():
                fct = t[:, None, None, None]
                transformed_image = (1-fct)*clean + fct*dirty
                predicted_peak = model(transformed_image, labels, t)
                
                loss = mse(clean, predicted_peak)
    
            valid_loss += loss.detach().item() * dirty.size(0)
            psnr_val += psnr(predicted_peak, clean, torch.max(predicted_peak)).detach()

        
    # Logging and saving
    if epoch % 5 == 0:
        ema_sampled_images = diffusion.sample(ema_model, x=dirty[0].reshape(1, 1, args.image_size, args.image_size), labels=None, n=1, steps=50)
        save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema_cond.png"))
        true_img = dirty[0].reshape(1, args.image_size, args.image_size).permute(1, 2, 0).cpu().numpy()
        gt_peak = clean[0].reshape(1, args.image_size, args.image_size).permute(1, 2, 0).cpu().numpy()
        ema_samp = ema_sampled_images[0].permute(1, 2, 0).cpu().numpy()
        # Create a figure with two subplots
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # Plot the original image in the first subplot
        ax1.imshow(true_img, origin='lower')
        ax1.set_title('Dirty Image')

        # Plot the EMA sampled image in the second subplot
        ax2.imshow(gt_peak, origin='lower')
        ax2.set_title('True Clean Image')

        ax3.imshow(ema_samp, origin='lower')
        ax3.set_title('Predicted Clean Image')
        
        plt.tight_layout()

        # Show the plot
        plt.show()

    plt.close()
    
    if min_valid_loss > valid_loss:
        logging.info(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
    
    # Saving State Dict
    torch.save(model.state_dict(), os.path.join("models", args.run_name, "ckpt_test_cond.pt"))
    torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, "ema_ckpt_cond.pt"))
