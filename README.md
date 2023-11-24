# Indi Model Training Repository

## Introduction
This repository contains the code for training the Indi model, an image processing model designed for image to image translation. This is a personal implementation of the Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration (InDI) paper by Delbracio et al 2023. The code is organized into three main files:
- `training.py`: The main script for training the model.
- `indi_diffusion.py`: Contains the Indi class and utility functions.
- `modules.py`: Includes essential modules and classes used in the model.

## Getting Started

### Prerequisites
List any prerequisites needed to run your project, such as Python version, libraries, etc. For example:
- Python 3.8+
- PyTorch
- PIL
- NumPy
- tqdm

### Installation
Provide steps to install any necessary libraries or set up the environment. For example:
```bash
pip install torch numpy pillow tqdm
```

### Results and Output
This implementation gives you an idea on how the InDI architeture work and it is made up for image to image translation tasks. For more information about it read the paper 
https://doi.org/10.48550/arXiv.2303.11435

### Contact
Contact me at francesco.ramunno@fhnw.ch

### References
- https://doi.org/10.48550/arXiv.2303.11435
- https://github.com/dome272/Diffusion-Models-pytorch
