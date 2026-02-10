"""
Stable Diffusion Pipeline Setup

This file initializes the Stable Diffusion img2img pipeline and
configures it based on available hardware (GPU or CPU).

Several memory optimisations are applied to ensure the model
can run on low-VRAM environments such as Colab.
"""

import warnings

# Must be set before torch import
warnings.filterwarnings(
    "ignore",
    message=r".*torch_dtype.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*CUDA is not available.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.amp\.autocast_mode",
)

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers.utils import logging as t_logging
from diffusers.utils import logging as d_logging

t_logging.set_verbosity_error()
d_logging.set_verbosity_error()

# ================================
# Device detection
# ================================
use_cuda = torch.cuda.is_available()

use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

# ================================
# Load Stable Diffusion pipeline
# ================================
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    # Use half precision on GPU to reduce memory usage
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    # Disable safety checker for faster inference
    safety_checker=None,
)

# ================================
# Hardware-specific optimisations
# ================================
if use_cuda:
    # Offloads parts of the model to CPU, reduces GPU memory usage
    pipe.enable_model_cpu_offload()

    # Use channels_last memory format for improved performance
    pipe.unet.to(memory_format=torch.channels_last)
elif use_mps:
    # Move model to MPS device (Apple Silicon)
    pipe = pipe.to("mps")

    # Enable attention slicing to reduce memory usage on MPS
    pipe.enable_attention_slicing(1)
else:
    # Fallback to CPU if CUDA GPU is not available
    pipe = pipe.to("cpu")


print("SD device:", "CUDA" if use_cuda else "MPS" if use_mps else "CPU")
